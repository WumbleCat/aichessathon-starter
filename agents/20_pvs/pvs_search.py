"""PVS selective search for 20_pvs (numba).

    iterative deepening (Python driver, class Searcher)
      -> aspiration window
      -> negamax PVS                      (negamax)
           -> transposition table         (tt_probe / tt_store)
           -> null-move pruning
           -> reverse futility pruning, razoring
           -> futility pruning, late-move pruning
           -> late-move reductions
           -> check extension
           -> move ordering: TT move, good captures (SEE), promotions, killers,
              history, bad captures       (score_moves / pick_move)
           -> quiescence search           (qsearch, stand pat + delta + SEE)
      -> best move from the last completed depth

Every heuristic has an integer toggle in the ``params`` array (see P_*), so A/B games can
disable it. All search state lives in numpy arrays passed to the jitted functions.
"""

from __future__ import annotations

import time

import numpy as np
from numba import boolean, float64, int64, njit, objmode, void

from pvs_board import (
    EMPTY,
    FLAG_CAPTURE,
    FLAG_EP,
    KING,
    MAX_PLY,
    MOVE_BUF,
    PAWN,
    QUEEN,
    ST_EP,
    ST_HALF,
    ST_HASH,
    ST_PLY,
    ST_SIDE,
    A1,
    A2,
    Position,
    attacked,
    attackers_to,
    bishop_attacks,
    gen_moves,
    king_square,
    lsb,
    make_move,
    make_null,
    move_to_uci,
    rook_attacks,
    unmake_move,
    unmake_null,
)
from pvs_eval import E_LMR, ETAB, SEE_VALUE, evaluate

# ------------------------------------------------------------------------ constants

MATE = 32000
MATE_BOUND = MATE - 2 * MAX_PLY  # scores beyond this are mates
INF = MATE + 1
TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
FLAG_NONE, FLAG_EXACT, FLAG_LOWER, FLAG_UPPER = 0, 1, 2, 3

# search info slots
(
    S_NODES,
    S_QNODES,
    S_STOP,
    S_SELDEPTH,
    S_TT_HITS,
    S_BETA_CUTS,
    S_FIRST_CUTS,
    S_NULL_CUTS,
    S_LMR_RESEARCH,
    S_ROOT_BEST,
    S_ROOT_SCORE,
    S_ROOT_DONE,
    S_AGE,
    S_REP_LEN,
    S_NODE_LIMIT,
    S_TT_STORES,
    S_ROOT_MOVES,
) = range(17)
S_SIZE = 24

# feature toggles / margins
(
    P_TT,
    P_NULL,
    P_RFP,
    P_RAZOR,
    P_FUTILITY,
    P_LMP,
    P_LMR,
    P_CHECK_EXT,
    P_SEE_ORDER,
    P_SEE_PRUNE_Q,
    P_SEE_PRUNE_MAIN,
    P_DELTA,
    P_KILLERS,
    P_HISTORY,
    P_PVS,
    P_ASPIRATION,
    P_IID,
    P_QS_TT,
    P_RFP_MARGIN,
    P_FUT_MARGIN,
    P_DELTA_MARGIN,
) = range(21)
P_SIZE = 32


def default_params() -> np.ndarray:
    p = np.ones(P_SIZE, dtype=np.int64)
    p[P_RFP_MARGIN] = 85
    p[P_FUT_MARGIN] = 110
    p[P_DELTA_MARGIN] = 150
    return p


# --------------------------------------------------------------------------- clock


@njit(float64(), cache=False)
def now() -> float:
    with objmode(t="float64"):
        t = time.perf_counter()
    return t


@njit(boolean(A1, float64[::1]), cache=False)
def check_stop(sinfo, sf) -> bool:
    if sinfo[S_STOP]:
        return True
    limit = sinfo[S_NODE_LIMIT]
    if limit > 0 and sinfo[S_NODES] >= limit:
        sinfo[S_STOP] = 1
        return True
    if (sinfo[S_NODES] & 2047) == 0 and sf[0] > 0.0 and now() > sf[0]:
        sinfo[S_STOP] = 1
        return True
    return False


# ------------------------------------------------------------------ transposition


@njit(int64(int64, int64, int64, int64, int64), cache=False)
def tt_pack(move: int, score: int, depth: int, flag: int, age: int) -> int:
    return (
        (move & 0x0FFFFFFF)
        | ((score + 32768) << 28)
        | ((depth + 8) << 44)
        | (flag << 52)
        | ((age & 0xFF) << 54)
    )


@njit(void(A1, A1, int64, int64, int64, int64, int64, int64), cache=False)
def tt_store(tt_keys, tt_data, key, move, score, depth, flag, age) -> None:
    idx = key & TT_MASK
    old = tt_data[idx]
    if tt_keys[idx] == key:
        # same position: keep the old move when the new entry has none
        if move == 0:
            move = old & 0x0FFFFFFF
        old_depth = ((old >> 44) & 0xFF) - 8
        if flag != FLAG_EXACT and old_depth > depth + 2 and ((old >> 54) & 0xFF) == (age & 0xFF):
            # keep the deeper entry but refresh its move
            tt_data[idx] = (old & ~0x0FFFFFFF) | (move & 0x0FFFFFFF)
            return
    else:
        old_depth = ((old >> 44) & 0xFF) - 8
        old_age = (old >> 54) & 0xFF
        if old != 0 and old_age == (age & 0xFF) and old_depth > depth + 3:
            return  # protect a much deeper entry from this search
    tt_keys[idx] = key
    tt_data[idx] = tt_pack(move, score, depth, flag, age)


@njit(int64(A1, A1, int64), cache=False)
def tt_probe(tt_keys, tt_data, key) -> int:
    idx = key & TT_MASK
    if tt_keys[idx] == key:
        return tt_data[idx]
    return 0


# ------------------------------------------------------------ static exchange eval


@njit(int64(A1, A1, A1, A1, int64, A1), cache=False)
def see(tab, bb, occ, sq, m: int, gain) -> int:
    """Material outcome (centipawns) of the capture sequence started by move ``m``."""
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    attacker = (m >> 20) & 15
    captured = (m >> 24) & 15
    side = (attacker - 1) // 6
    all_occ = occ[2]
    occ_sim = all_occ ^ (1 << frm)
    if m & FLAG_EP:
        cap_sq = to - 8 if side == 0 else to + 8
        occ_sim ^= 1 << cap_sq
    on_square = SEE_VALUE[attacker]
    gain[0] = SEE_VALUE[captured]
    if promo:
        gain[0] += SEE_VALUE[promo] - SEE_VALUE[PAWN]
        on_square = SEE_VALUE[promo]
    side = 1 - side
    d = 0
    attackers = attackers_to(tab, bb, occ_sim, to) & occ_sim
    while True:
        my = attackers & occ[side]
        if my == 0:
            break
        # least valuable attacker of ``side``
        pt = 0
        for cand in range(1, 7):
            b = my & bb[6 * side + cand]
            if b:
                pt = cand
                frm = lsb(b)
                break
        if pt == KING and (attackers & occ[1 - side]):
            break  # the king may not capture into an attacked square
        d += 1
        gain[d] = on_square - gain[d - 1]
        if gain[d] < 0 and -gain[d - 1] < 0:
            break  # neither side can improve: speculative cut-off
        on_square = SEE_VALUE[pt]
        occ_sim ^= 1 << frm
        attackers = attackers_to(tab, bb, occ_sim, to) & occ_sim
        side = 1 - side
        if d >= 30:
            break
    while d > 0:
        if -gain[d] < gain[d - 1]:
            gain[d - 1] = -gain[d]
        d -= 1
    return gain[0]


# ------------------------------------------------------------------ move ordering

SCORE_TT = 1 << 30
SCORE_GOOD_CAPTURE = 1 << 28
SCORE_KILLER = 1 << 27
SCORE_BAD_CAPTURE = -(1 << 20)
HISTORY_MAX = 1 << 14


@njit(
    void(A1, A1, A1, A1, A1, A1, A1, int64, int64, int64, A1, A1, int64, A1, A1),
    cache=False,
)
def score_moves(
    tab, bb, occ, sq, st, moves, scores, base, n, tt_move, killers, history, ply, params, gain
) -> None:
    side = st[ST_SIDE]
    k0 = killers[ply * 2]
    k1 = killers[ply * 2 + 1]
    hist_base = side * 4096
    for i in range(base, base + n):
        m = moves[i]
        if m == tt_move:
            scores[i] = SCORE_TT
            continue
        captured = (m >> 24) & 15
        promo = (m >> 12) & 7
        if captured or promo:
            victim = SEE_VALUE[captured]
            if promo:
                if promo == QUEEN:
                    victim += SEE_VALUE[QUEEN]
                else:
                    scores[i] = SCORE_BAD_CAPTURE - 1000 + victim
                    continue
            attacker = (m >> 20) & 15
            if params[P_SEE_ORDER] and captured and SEE_VALUE[captured] < SEE_VALUE[attacker]:
                if see(tab, bb, occ, sq, m, gain) < 0:
                    scores[i] = SCORE_BAD_CAPTURE + victim
                    continue
            scores[i] = SCORE_GOOD_CAPTURE + victim * 16 - ((attacker - 1) % 6)
        elif params[P_KILLERS] and m == k0:
            scores[i] = SCORE_KILLER
        elif params[P_KILLERS] and m == k1:
            scores[i] = SCORE_KILLER - 1
        elif params[P_HISTORY]:
            scores[i] = history[hist_base + (m & 4095)]
        else:
            scores[i] = 0


@njit(int64(A1, A1, int64, int64, int64), cache=False)
def pick_move(moves, scores, base, n, i) -> int:
    """Selection sort step: move the best remaining move to slot i and return it."""
    best = i
    best_score = scores[base + i]
    for j in range(i + 1, n):
        if scores[base + j] > best_score:
            best_score = scores[base + j]
            best = j
    if best != i:
        tm = moves[base + i]
        moves[base + i] = moves[base + best]
        moves[base + best] = tm
        ts = scores[base + i]
        scores[base + i] = scores[base + best]
        scores[base + best] = ts
    return moves[base + i]


@njit(void(A1, int64, int64, int64), cache=False)
def history_update(history, idx, bonus, is_good) -> None:
    h = history[idx]
    if is_good:
        h += bonus - h * bonus // HISTORY_MAX
    else:
        h -= bonus + h * bonus // HISTORY_MAX
    history[idx] = h


# ---------------------------------------------------------------------- quiescence


@njit(
    int64(A1, A1, A1, A1, A1, A1, A2, A1, A1, A1, A1, A1, A1, A1, A1, float64[::1], A1, A1,
          int64, int64, int64),
    cache=False,
)
def qsearch(
    tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers, history,
    rep, sinfo, sf, params, gain, alpha, beta, ply,
) -> int:
    sinfo[S_NODES] += 1
    sinfo[S_QNODES] += 1
    if check_stop(sinfo, sf):
        return 0
    if ply > sinfo[S_SELDEPTH]:
        sinfo[S_SELDEPTH] = ply
    side = st[ST_SIDE]
    checked = attacked(tab, bb, occ[2], king_square(bb, side), 1 - side)
    if ply >= MAX_PLY - 1:
        return evaluate(tab, etab, bb, occ, st)

    tt_move = 0
    key = st[ST_HASH]
    if params[P_QS_TT]:
        entry = tt_probe(tt_keys, tt_data, key)
        if entry != 0:
            sinfo[S_TT_HITS] += 1
            tscore = ((entry >> 28) & 0xFFFF) - 32768
            if tscore > MATE_BOUND:
                tscore -= ply
            elif tscore < -MATE_BOUND:
                tscore += ply
            tflag = (entry >> 52) & 3
            if tflag == FLAG_EXACT:
                return tscore
            if tflag == FLAG_LOWER and tscore >= beta:
                return tscore
            if tflag == FLAG_UPPER and tscore <= alpha:
                return tscore
            tt_move = entry & 0x0FFFFFFF

    best = -INF
    if not checked:
        stand = evaluate(tab, etab, bb, occ, st)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand

    base = ply * MOVE_BUF
    n = gen_moves(tab, bb, occ, sq, st, moves, base, not checked)
    score_moves(
        tab, bb, occ, sq, st, moves, scores, base, n, tt_move, killers, history, ply, params, gain
    )
    legal = 0
    best_move = 0
    alpha_orig = alpha
    for i in range(n):
        m = pick_move(moves, scores, base, n, i)
        if not checked:
            captured = (m >> 24) & 15
            promo = (m >> 12) & 7
            if params[P_DELTA] and promo == 0:
                if best + SEE_VALUE[captured] + params[P_DELTA_MARGIN] <= alpha:
                    continue
            if params[P_SEE_PRUNE_Q] and scores[base + i] < 0 and promo == 0:
                continue  # losing capture (SEE < 0) or under-promotion
        make_move(tab, bb, occ, sq, st, undo, m)
        mover = 1 - st[ST_SIDE]
        if attacked(tab, bb, occ[2], king_square(bb, mover), st[ST_SIDE]):
            unmake_move(bb, occ, sq, st, undo)
            continue
        legal += 1
        score = -qsearch(
            tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers, history,
            rep, sinfo, sf, params, gain, -beta, -alpha, ply + 1,
        )
        unmake_move(bb, occ, sq, st, undo)
        if sinfo[S_STOP]:
            return 0
        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if checked and legal == 0:
        return -MATE + ply
    if params[P_QS_TT]:
        flag = FLAG_UPPER
        if best >= beta:
            flag = FLAG_LOWER
        elif best > alpha_orig:
            flag = FLAG_EXACT
        tscore = best
        if tscore > MATE_BOUND:
            tscore += ply
        elif tscore < -MATE_BOUND:
            tscore -= ply
        tt_store(tt_keys, tt_data, key, best_move, tscore, 0, flag, sinfo[S_AGE])
    return best


# ------------------------------------------------------------------------- negamax


@njit(
    int64(A1, A1, A1, A1, A1, A1, A2, A1, A1, A1, A1, A1, A1, A1, A1, float64[::1], A1, A1,
          int64, int64, int64, int64, boolean),
    cache=False,
)
def negamax(
    tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers, history,
    rep, sinfo, sf, params, gain, depth, alpha, beta, ply, can_null,
) -> int:
    if sinfo[S_STOP]:
        return 0
    pv_node = beta - alpha > 1
    root = ply == 0
    key = st[ST_HASH]
    rep_pos = sinfo[S_REP_LEN] + ply
    rep[rep_pos] = key

    if not root:
        # draw by repetition (any earlier occurrence, game history included) or 50 moves
        half = st[ST_HALF]
        if half >= 100:
            return 0
        i = rep_pos - 2
        stop_at = rep_pos - half
        if stop_at < 0:
            stop_at = 0
        while i >= stop_at:
            if rep[i] == key:
                return 0
            i -= 2
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    side = st[ST_SIDE]
    checked = attacked(tab, bb, occ[2], king_square(bb, side), 1 - side)
    if checked and params[P_CHECK_EXT] and ply < MAX_PLY - 8:
        depth += 1

    if depth <= 0:
        return qsearch(
            tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers, history,
            rep, sinfo, sf, params, gain, alpha, beta, ply,
        )
    if ply >= MAX_PLY - 1:
        return evaluate(tab, etab, bb, occ, st)

    sinfo[S_NODES] += 1
    if check_stop(sinfo, sf):
        return 0

    # ---- transposition table
    tt_move = 0
    tt_hit = False
    tt_score = 0
    tt_flag = FLAG_NONE
    tt_depth = -100
    if params[P_TT]:
        entry = tt_probe(tt_keys, tt_data, key)
        if entry != 0:
            sinfo[S_TT_HITS] += 1
            tt_hit = True
            tt_move = entry & 0x0FFFFFFF
            tt_score = ((entry >> 28) & 0xFFFF) - 32768
            if tt_score > MATE_BOUND:
                tt_score -= ply
            elif tt_score < -MATE_BOUND:
                tt_score += ply
            tt_depth = ((entry >> 44) & 0xFF) - 8
            tt_flag = (entry >> 52) & 3
            if not pv_node and tt_depth >= depth:
                if tt_flag == FLAG_EXACT:
                    return tt_score
                if tt_flag == FLAG_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == FLAG_UPPER and tt_score <= alpha:
                    return tt_score

    # ---- static evaluation and forward pruning at non-PV, non-check nodes
    static_eval = 0
    if not checked:
        static_eval = evaluate(tab, etab, bb, occ, st)
        if tt_hit:
            # a bound from the table refines the static estimate
            if tt_flag == FLAG_EXACT:
                static_eval = tt_score
            elif tt_flag == FLAG_LOWER and tt_score > static_eval:
                static_eval = tt_score
            elif tt_flag == FLAG_UPPER and tt_score < static_eval:
                static_eval = tt_score

    if not pv_node and not checked and abs(beta) < MATE_BOUND:
        # reverse futility pruning: far above beta at low depth
        if params[P_RFP] and depth <= 7 and static_eval - params[P_RFP_MARGIN] * depth >= beta:
            return static_eval
        # razoring: hopeless at depth 1-2, let quiescence decide
        if params[P_RAZOR] and depth <= 2 and static_eval + 200 * depth <= alpha:
            qs = qsearch(
                tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
                history, rep, sinfo, sf, params, gain, alpha, beta, ply,
            )
            if qs <= alpha:
                return qs
        # null move pruning
        if params[P_NULL] and can_null and depth >= 2 and static_eval >= beta:
            base_side = 6 * side
            non_pawn = bb[base_side + 2] | bb[base_side + 3] | bb[base_side + 4] | bb[base_side + 5]
            if non_pawn:
                r = 3 + depth // 4
                if static_eval - beta > 200:
                    r += 1
                make_null(tab, st, undo)
                score = -negamax(
                    tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
                    history, rep, sinfo, sf, params, gain, depth - 1 - r, -beta, -beta + 1,
                    ply + 1, False,
                )
                unmake_null(st, undo)
                if sinfo[S_STOP]:
                    return 0
                if score >= beta:
                    sinfo[S_NULL_CUTS] += 1
                    if score > MATE_BOUND:
                        score = beta
                    return score

    # ---- internal iterative deepening when no hash move at a PV node
    if params[P_IID] and pv_node and tt_move == 0 and depth >= 5:
        negamax(
            tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
            history, rep, sinfo, sf, params, gain, depth - 2, alpha, beta, ply, can_null,
        )
        if sinfo[S_STOP]:
            return 0
        entry = tt_probe(tt_keys, tt_data, key)
        if entry != 0:
            tt_move = entry & 0x0FFFFFFF

    # ---- move loop
    base = ply * MOVE_BUF
    n = gen_moves(tab, bb, occ, sq, st, moves, base, False)
    score_moves(
        tab, bb, occ, sq, st, moves, scores, base, n, tt_move, killers, history, ply, params, gain
    )
    futile = (
        params[P_FUTILITY] != 0
        and not pv_node
        and not checked
        and depth <= 4
        and static_eval + params[P_FUT_MARGIN] * depth + 50 <= alpha
    )
    lmp_limit = 3 + depth * depth * 2
    best = -INF
    best_move = 0
    alpha_orig = alpha
    legal = 0
    hist_base = side * 4096
    for i in range(n):
        m = pick_move(moves, scores, base, n, i)
        captured = (m >> 24) & 15
        promo = (m >> 12) & 7
        quiet = captured == 0 and promo == 0
        mscore = scores[base + i]

        # forward pruning of late quiet moves at shallow depth (never the first move)
        if legal > 0 and quiet and not pv_node and not checked and abs(alpha) < MATE_BOUND:
            if futile and mscore < SCORE_KILLER - 1:
                continue
            if params[P_LMP] and depth <= 3 and legal >= lmp_limit and mscore < SCORE_KILLER - 1:
                continue
        # prune clearly losing captures at shallow depth
        if (
            legal > 0
            and params[P_SEE_PRUNE_MAIN]
            and not pv_node
            and not checked
            and depth <= 3
            and mscore <= SCORE_BAD_CAPTURE + 1000
            and mscore > SCORE_BAD_CAPTURE - 1000
        ):
            continue

        make_move(tab, bb, occ, sq, st, undo, m)
        if attacked(tab, bb, occ[2], king_square(bb, side), 1 - side):
            unmake_move(bb, occ, sq, st, undo)
            continue
        legal += 1
        gives_check = attacked(tab, bb, occ[2], king_square(bb, 1 - side), side)
        new_depth = depth - 1

        if legal == 1 or not params[P_PVS]:
            score = -negamax(
                tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
                history, rep, sinfo, sf, params, gain, new_depth, -beta, -alpha, ply + 1, True,
            )
        else:
            reduction = 0
            if (
                params[P_LMR]
                and depth >= 3
                and quiet
                and not checked
                and not gives_check
                and legal > 2
            ):
                reduction = etab[E_LMR + (min(depth, 63) << 6) + min(legal, 63)]
                if pv_node:
                    reduction -= 1
                if mscore >= SCORE_KILLER - 1:
                    reduction -= 1
                if history[hist_base + (m & 4095)] > HISTORY_MAX // 2:
                    reduction -= 1
                if reduction < 0:
                    reduction = 0
                if reduction > new_depth - 1:
                    reduction = new_depth - 1
                    if reduction < 0:
                        reduction = 0
            score = -negamax(
                tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
                history, rep, sinfo, sf, params, gain, new_depth - reduction, -alpha - 1, -alpha,
                ply + 1, True,
            )
            if reduction > 0 and score > alpha and not sinfo[S_STOP]:
                sinfo[S_LMR_RESEARCH] += 1
                score = -negamax(
                    tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
                    history, rep, sinfo, sf, params, gain, new_depth, -alpha - 1, -alpha,
                    ply + 1, True,
                )
            if score > alpha and score < beta and not sinfo[S_STOP]:
                score = -negamax(
                    tab, etab, bb, occ, sq, st, undo, moves, scores, tt_keys, tt_data, killers,
                    history, rep, sinfo, sf, params, gain, new_depth, -beta, -alpha, ply + 1,
                    True,
                )
        unmake_move(bb, occ, sq, st, undo)
        if sinfo[S_STOP]:
            return 0
        if root:
            sinfo[S_ROOT_DONE] += 1

        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                if root:
                    sinfo[S_ROOT_BEST] = m
                    sinfo[S_ROOT_SCORE] = score
                if alpha >= beta:
                    sinfo[S_BETA_CUTS] += 1
                    if legal == 1:
                        sinfo[S_FIRST_CUTS] += 1
                    if quiet:
                        if params[P_KILLERS] and killers[ply * 2] != m:
                            killers[ply * 2 + 1] = killers[ply * 2]
                            killers[ply * 2] = m
                        if params[P_HISTORY]:
                            bonus = depth * depth
                            if bonus > 400:
                                bonus = 400
                            history_update(history, hist_base + (m & 4095), bonus, 1)
                            for j in range(i):
                                pm = moves[base + j]
                                if ((pm >> 24) & 15) == 0 and ((pm >> 12) & 7) == 0:
                                    history_update(history, hist_base + (pm & 4095), bonus, 0)
                    break

    if legal == 0:
        if checked:
            return -MATE + ply
        return 0

    if params[P_TT]:
        flag = FLAG_UPPER
        if best >= beta:
            flag = FLAG_LOWER
        elif best > alpha_orig:
            flag = FLAG_EXACT
        tscore = best
        if tscore > MATE_BOUND:
            tscore += ply
        elif tscore < -MATE_BOUND:
            tscore -= ply
        tt_store(tt_keys, tt_data, key, best_move, tscore, depth, flag, sinfo[S_AGE])
        sinfo[S_TT_STORES] += 1
    return best


# ------------------------------------------------------------------------ driver


class Searcher:
    """Iterative deepening with aspiration windows. One instance lives for a whole game."""

    def __init__(self, params: np.ndarray | None = None) -> None:
        self.params = default_params() if params is None else params
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_data = np.zeros(TT_SIZE, dtype=np.int64)
        self.killers = np.zeros((MAX_PLY + 8) * 2, dtype=np.int64)
        self.history = np.zeros(2 * 4096, dtype=np.int64)
        self.rep = np.zeros(1024 + MAX_PLY + 8, dtype=np.int64)
        self.scores = np.zeros((MAX_PLY + 8) * MOVE_BUF, dtype=np.int64)
        self.sinfo = np.zeros(S_SIZE, dtype=np.int64)
        self.sf = np.zeros(2, dtype=np.float64)
        self.gain = np.zeros(64, dtype=np.int64)
        self.etab = ETAB
        self.age = 0
        self.last_info: dict = {}

    def clear(self) -> None:
        self.tt_keys[:] = 0
        self.tt_data[:] = 0
        self.killers[:] = 0
        self.history[:] = 0

    def search(
        self,
        pos: Position,
        time_budget: float | None = None,
        max_depth: int = 64,
        node_limit: int = 0,
        history_keys: list[int] | None = None,
        verbose: bool = False,
    ) -> tuple[int, int, int, dict]:
        """Return (move, score, depth, info). ``move`` is 0 only when there is no legal move."""
        start = time.perf_counter()
        self.age = (self.age + 1) & 0xFF
        self.killers[:] = 0
        self.history //= 2
        sinfo = self.sinfo
        sinfo[:] = 0
        sinfo[S_AGE] = self.age
        keys = history_keys or []
        keys = keys[-1000:]
        for i, k in enumerate(keys):
            self.rep[i] = k
        sinfo[S_REP_LEN] = len(keys)
        sinfo[S_NODE_LIMIT] = node_limit
        self.sf[0] = 0.0  # depth 1 always completes

        legal = pos.legal_moves()
        if not legal:
            return 0, 0, 0, {"nodes": 0}
        best_move = legal[0]
        best_score = 0
        completed = 0
        args = (
            pos.tab, self.etab, pos.bb, pos.occ, pos.sq, pos.st, pos.undo, pos.moves,
            self.scores, self.tt_keys, self.tt_data, self.killers, self.history, self.rep,
            sinfo, self.sf, self.params, self.gain,
        )
        depth = 1
        window = 20
        alpha, beta = -INF, INF
        while depth <= max_depth:
            sinfo[S_ROOT_BEST] = 0
            sinfo[S_ROOT_DONE] = 0
            sinfo[S_ROOT_SCORE] = -INF
            if depth >= 2 and time_budget is not None:
                self.sf[0] = start + time_budget
            if not self.params[P_ASPIRATION] or depth < 4:
                alpha, beta = -INF, INF
            score = negamax(*args, depth, alpha, beta, 0, True)
            pos.st[ST_PLY] = 0  # a timeout unwinds without popping
            if sinfo[S_STOP]:
                # take a root move only if it was fully searched and beat the old best
                if sinfo[S_ROOT_BEST] != 0 and sinfo[S_ROOT_SCORE] > -INF:
                    if completed == 0 or sinfo[S_ROOT_BEST] != best_move:
                        best_move = int(sinfo[S_ROOT_BEST])
                        best_score = int(sinfo[S_ROOT_SCORE])
                break
            if self.params[P_ASPIRATION] and depth >= 4 and (score <= alpha or score >= beta):
                # aspiration failure: widen and repeat the same depth
                if score <= alpha:
                    alpha = max(-INF, score - window)
                else:
                    beta = min(INF, score + window)
                window *= 3
                if window > 1200:
                    alpha, beta = -INF, INF
                if verbose:
                    print(f"depth {depth} aspiration fail score {score} window {window}")
                continue
            completed = depth
            best_score = int(score)
            if sinfo[S_ROOT_BEST] != 0:
                best_move = int(sinfo[S_ROOT_BEST])
            elapsed = time.perf_counter() - start
            if verbose:
                print(
                    f"depth {depth} seldepth {sinfo[S_SELDEPTH]} score {best_score} "
                    f"move {move_to_uci(best_move)} nodes {sinfo[S_NODES]} "
                    f"qnodes {sinfo[S_QNODES]} time {elapsed:.2f}s "
                    f"pv {self.pv_string(pos)}"
                )
            if abs(best_score) >= MATE_BOUND and depth >= 4:
                break
            if time_budget is not None and elapsed > 0.55 * time_budget:
                break  # the next iteration would very likely not finish
            window = 20
            alpha, beta = best_score - window, best_score + window
            depth += 1
        elapsed = time.perf_counter() - start
        info = {
            "depth": completed,
            "seldepth": int(sinfo[S_SELDEPTH]),
            "nodes": int(sinfo[S_NODES]),
            "qnodes": int(sinfo[S_QNODES]),
            "nps": int(sinfo[S_NODES] / elapsed) if elapsed > 0 else 0,
            "tt_hits": int(sinfo[S_TT_HITS]),
            "beta_cuts": int(sinfo[S_BETA_CUTS]),
            "first_cuts": int(sinfo[S_FIRST_CUTS]),
            "null_cuts": int(sinfo[S_NULL_CUTS]),
            "lmr_research": int(sinfo[S_LMR_RESEARCH]),
            "time": elapsed,
            "score": best_score,
        }
        self.last_info = info
        return best_move, best_score, completed, info

    def pv_string(self, pos: Position, max_len: int = 10) -> str:
        """Follow TT moves from the current position (which must be at ply 0)."""
        pv = []
        pushed = 0
        seen = set()
        for _ in range(max_len):
            key = int(pos.st[ST_HASH])
            if key in seen:
                break
            seen.add(key)
            entry = int(tt_probe(self.tt_keys, self.tt_data, key))
            if entry == 0:
                break
            m = entry & 0x0FFFFFFF
            if m == 0 or m not in pos.legal_moves():
                break
            pv.append(move_to_uci(m))
            pos.push(m)
            pushed += 1
        for _ in range(pushed):
            pos.pop()
        return " ".join(pv)
