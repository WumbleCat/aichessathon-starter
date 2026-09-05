"""Alpha-beta search (PVS, TT, killers, history, null move, LMR, futility, quiescence).

All hot code is numba-compiled.  State lives in numpy arrays owned by :class:`Searcher`:

    P, undo, mv/ms     position, undo stack, per-ply move lists and ordering scores (cboard)
    acc                NNUE accumulators per ply (nnue)
    net                (W1, B1, W2, B2) quantised weights
    tt_key / tt_val    transposition table, val packs move | score | depth | flag
    killers, history   move-ordering heuristics
    keys               zobrist key of each ply on the current path (repetition detection)
    ghist              zobrist keys of positions already seen in the game
    ctl / ctf          integer / float control and statistics slots (indices C_* / F_*)
"""

from __future__ import annotations

import time

import numpy as np
from numba import objmode

import cboard as cb
import nnue
from jitconf import jit

MATE = 30000
MATE_BOUND = MATE - 512
INF = 32000
MAX_DEPTH = 64
TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
FLAG_EXACT, FLAG_LOWER, FLAG_UPPER = 1, 2, 3
NODES_PER_CLOCK_CHECK = 1024

# ctl slots
C_NODES, C_QNODES, C_ABORT, C_NODE_LIMIT, C_SELDEPTH, C_TT_HITS, C_NEXT_CHECK = 0, 1, 2, 3, 4, 5, 6
C_GHIST_N, C_ROOT_MOVE, C_ROOT_SCORE, C_USE_NNUE, C_ROOT_DEPTH, C_PV_LEN = 7, 8, 9, 10, 11, 12
C_ROOT_MOVES_DONE, C_NULL_PLY = 13, 14
CTL_SIZE = 32
F_DEADLINE = 0

# material / PSQT fallback evaluation (used when C_USE_NNUE == 0, mainly for tests and A/B)
PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0, 100, 320, 330, 500, 900, 0], dtype=np.int64)
MVV = np.array([0, 100, 320, 330, 500, 900, 2000], dtype=np.int64)

_PST = {
    cb.PAWN: [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    cb.KNIGHT: [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
    ],
    cb.BISHOP: [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
    ],
    cb.ROOK: [
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    cb.QUEEN: [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        5,
        0,
        0,
        0,
        0,
        -10,
        -10,
        5,
        5,
        5,
        5,
        5,
        0,
        -10,
        0,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
    ],
    cb.KING: [
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
        20,
        20,
        0,
        0,
        0,
        0,
        20,
        20,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        -20,
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -20,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
    ],
}
PSQT = np.zeros((13, 64), dtype=np.int64)
for _t, _tab in _PST.items():
    for _sq in range(64):
        PSQT[_t, _sq] = PIECE_VALUE[_t] + _tab[_sq]
        PSQT[_t + 6, _sq ^ 56] = -(PIECE_VALUE[_t] + _tab[_sq])

LMR_TABLE = np.zeros((MAX_DEPTH + 1, 64), dtype=np.int64)
for _d in range(1, MAX_DEPTH + 1):
    for _m in range(1, 64):
        LMR_TABLE[_d, _m] = int(0.75 + np.log(_d) * np.log(_m) / 2.25)


class SearchTimeout(Exception):
    pass


# ----------------------------------------------------------------------------- helpers


@jit
def psqt_eval(P):  # type: ignore[no-untyped-def]
    s = 0
    for sq in range(64):
        pc = P[sq]
        if pc != 0:
            s += PSQT[pc, sq]
    return s if P[cb.SIDE] == 0 else -s


@jit
def static_eval(P, acc, ply, net, ctl):  # type: ignore[no-untyped-def]
    if ctl[C_USE_NNUE] != 0:
        return nnue.evaluate(acc, ply, P[cb.SIDE], net[2], net[3])
    return psqt_eval(P)


@jit
def is_draw(P, keys, ply, ghist, ctl):  # type: ignore[no-untyped-def]
    """Repetition (path or game history), 50-move rule, bare/insufficient material."""
    half = P[cb.HALF]
    if half >= 100:
        return True
    key = P[cb.HASH]
    if half >= 4:
        n = ctl[C_GHIST_N]
        d = 4  # ply distance back; same side to move needs an even distance
        while d <= half:
            if d <= ply:
                k = keys[ply - d]
            else:
                j = n - 1 - (d - ply)  # ghist[n-1] is the root position (keys[0])
                if j < 0:
                    break
                k = ghist[j]
            if k == key:
                return True
            d += 2
    if (P[cb.BB + cb.WP] | P[cb.BB + cb.BP]) == 0:
        heavy = P[cb.BB + cb.WR] | P[cb.BB + cb.BR] | P[cb.BB + cb.WQ] | P[cb.BB + cb.BQ]
        if (heavy == 0) & (P[cb.NONPAWN] <= 1):
            return True
    return False


@jit
def tt_probe(tt_key, tt_val, key):  # type: ignore[no-untyped-def]
    idx = key & TT_MASK
    if tt_key[idx] == key:
        return tt_val[idx]
    return np.int64(0)


@jit
def tt_store(tt_key, tt_val, key, move, score, depth, flag):  # type: ignore[no-untyped-def]
    idx = key & TT_MASK
    # depth-preferred replacement unless the slot belongs to another position or is old
    if tt_key[idx] == key:
        old_depth = (tt_val[idx] >> 40) & 255
        if (old_depth > depth + 2) & (flag != FLAG_EXACT):
            return
    tt_key[idx] = key
    tt_val[idx] = (move & 0xFFFFF) | ((score + 32768) << 20) | (depth << 40) | (flag << 48)


@jit
def tt_move(val):  # type: ignore[no-untyped-def]
    return val & 0xFFFFF


@jit
def tt_score(val):  # type: ignore[no-untyped-def]
    return ((val >> 20) & 0xFFFF) - 32768


@jit
def tt_depth(val):  # type: ignore[no-untyped-def]
    return (val >> 40) & 255


@jit
def tt_flag(val):  # type: ignore[no-untyped-def]
    return (val >> 48) & 3


@jit
def score_to_tt(score, ply):  # type: ignore[no-untyped-def]
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@jit
def score_from_tt(score, ply):  # type: ignore[no-untyped-def]
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


@jit
def check_clock(ctl, ctf):  # type: ignore[no-untyped-def]
    nodes = ctl[C_NODES]
    if nodes >= ctl[C_NEXT_CHECK]:
        ctl[C_NEXT_CHECK] = nodes + NODES_PER_CLOCK_CHECK
        if (ctl[C_NODE_LIMIT] > 0) & (nodes >= ctl[C_NODE_LIMIT]):
            ctl[C_ABORT] = 1
        if ctf[F_DEADLINE] > 0.0:
            with objmode(now="float64"):
                now = time.perf_counter()
            if now >= ctf[F_DEADLINE]:
                ctl[C_ABORT] = 1


@jit
def score_moves(P, moves, scores, n, ttmv, killers, history, ply):  # type: ignore[no-untyped-def]
    side = P[cb.SIDE]
    for i in range(n):
        m = moves[i]
        if m == ttmv:
            scores[i] = 10_000_000
            continue
        flags = cb.mv_flags(m)
        promo = cb.mv_promo(m)
        if flags & cb.F_CAPTURE:
            victim = cb.PAWN if (flags & cb.F_EP) else cb.piece_type(P[cb.mv_to(m)])
            attacker = cb.piece_type(P[cb.mv_from(m)])
            scores[i] = 1_000_000 + MVV[victim] * 10 - attacker
            if promo == cb.QUEEN:
                scores[i] += 900
        elif promo != 0:
            scores[i] = 1_000_000 + (900 if promo == cb.QUEEN else -500)
        elif m == killers[ply, 0]:
            scores[i] = 900_000
        elif m == killers[ply, 1]:
            scores[i] = 800_000
        else:
            scores[i] = history[side, cb.mv_from(m), cb.mv_to(m)]


@jit
def pick_next(moves, scores, n, i):  # type: ignore[no-untyped-def]
    """Selection step: swap the best-scored remaining move into slot i."""
    best = i
    bs = scores[i]
    for j in range(i + 1, n):
        if scores[j] > bs:
            bs = scores[j]
            best = j
    if best != i:
        tm = moves[i]
        moves[i] = moves[best]
        moves[best] = tm
        ts = scores[i]
        scores[i] = scores[best]
        scores[best] = ts
    return moves[i]


@jit
def do_move(P, undo, ply, move, acc, net, ctl):  # type: ignore[no-untyped-def]
    if not cb.make_move(P, undo, ply, move):
        return False
    if ctl[C_USE_NNUE] != 0:
        nnue.update(acc, ply, P, net[0])
    return True


# ----------------------------------------------------------------------------- quiescence

# The search state travels as ONE tuple ``S`` (see ``Searcher.search``) so that numba's type
# inference sees one argument instead of fourteen: the compile of the recursive functions is
# dominated by inference over their arguments.  Indices into that tuple:
X_P, X_UNDO, X_MV, X_MS, X_ACC, X_NET, X_TT_KEY, X_TT_VAL = 0, 1, 2, 3, 4, 5, 6, 7
X_KILLERS, X_HISTORY, X_KEYS, X_GHIST, X_CTL, X_CTF = 8, 9, 10, 11, 12, 13


@jit
def qsearch(S, ply, alpha, beta):  # type: ignore[no-untyped-def]
    P = S[X_P]
    undo = S[X_UNDO]
    acc = S[X_ACC]
    net = S[X_NET]
    ctl = S[X_CTL]
    ctl[C_NODES] += 1
    ctl[C_QNODES] += 1
    if ply > ctl[C_SELDEPTH]:
        ctl[C_SELDEPTH] = ply
    check_clock(ctl, S[X_CTF])
    if ctl[C_ABORT] != 0:
        return 0
    if ply >= cb.MAX_PLY - 1:
        return static_eval(P, acc, ply, net, ctl)
    stand = static_eval(P, acc, ply, net, ctl)
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand
    moves = S[X_MV][ply]
    scores = S[X_MS][ply]
    n = cb.gen_moves(P, moves, cb.CAPTURES_ONLY)
    score_moves(P, moves, scores, n, np.int64(0), S[X_KILLERS], S[X_HISTORY], ply)
    keys = S[X_KEYS]
    best = stand
    for i in range(n):
        m = pick_next(moves, scores, n, i)
        # delta pruning: even winning the victim outright cannot raise alpha
        flags = cb.mv_flags(m)
        if cb.mv_promo(m) == 0:
            victim = cb.PAWN if (flags & cb.F_EP) else cb.piece_type(P[cb.mv_to(m)])
            if stand + MVV[victim] + 200 <= alpha:
                continue
        if not do_move(P, undo, ply, m, acc, net, ctl):
            cb.unmake_move(P, undo, ply)
            continue
        keys[ply + 1] = P[cb.HASH]
        score = -qsearch(S, ply + 1, -beta, -alpha)
        cb.unmake_move(P, undo, ply)
        if ctl[C_ABORT] != 0:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


# ----------------------------------------------------------------------------- main search


@jit
def negamax(S, ply, depth, alpha, beta):  # type: ignore[no-untyped-def]
    if depth <= 0:
        return qsearch(S, ply, alpha, beta)
    P = S[X_P]
    undo = S[X_UNDO]
    acc = S[X_ACC]
    net = S[X_NET]
    tt_key = S[X_TT_KEY]
    tt_val = S[X_TT_VAL]
    killers = S[X_KILLERS]
    history = S[X_HISTORY]
    keys = S[X_KEYS]
    ctl = S[X_CTL]
    ctl[C_NODES] += 1
    check_clock(ctl, S[X_CTF])
    if ctl[C_ABORT] != 0:
        return 0
    if ply > ctl[C_SELDEPTH]:
        ctl[C_SELDEPTH] = ply
    if ply >= cb.MAX_PLY - 1:
        return static_eval(P, acc, ply, net, ctl)

    pv_node = beta - alpha > 1
    root = ply == 0
    # a null move may not answer a null move (and the root never makes one)
    do_null = ctl[C_NULL_PLY] != ply
    if not root:
        if is_draw(P, keys, ply, S[X_GHIST], ctl):
            return 0
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    key = P[cb.HASH]
    val = tt_probe(tt_key, tt_val, key)
    ttmv = np.int64(0)
    if val != 0:
        ttmv = tt_move(val)
        if (not pv_node) & (tt_depth(val) >= depth):
            s = score_from_tt(tt_score(val), ply)
            f = tt_flag(val)
            if (
                (f == FLAG_EXACT)
                | ((f == FLAG_LOWER) & (s >= beta))
                | ((f == FLAG_UPPER) & (s <= alpha))
            ):
                ctl[C_TT_HITS] += 1
                return s

    incheck = cb.in_check(P)
    if incheck:
        depth += 1  # check extension

    stat = 0
    if not incheck:
        stat = static_eval(P, acc, ply, net, ctl)
        if not pv_node:
            # reverse futility pruning
            if (depth <= 6) & (stat - 90 * depth >= beta) & (beta > -MATE_BOUND):
                return stat
            # null move pruning
            if do_null & (depth >= 2) & (stat >= beta) & (P[cb.NONPAWN] >= 2):
                r = 3 + depth // 4 + min(3, (stat - beta) // 200)
                cb.make_null(P, undo, ply)
                if ctl[C_USE_NNUE] != 0:
                    nnue.copy_acc(acc, ply)
                keys[ply + 1] = P[cb.HASH]
                ctl[C_NULL_PLY] = ply + 1
                score = -negamax(S, ply + 1, depth - 1 - r, -beta, -beta + 1)
                ctl[C_NULL_PLY] = -1
                cb.unmake_null(P, undo, ply)
                if ctl[C_ABORT] != 0:
                    return 0
                if score >= beta:
                    if score >= MATE_BOUND:
                        score = beta
                    return score
            # razoring
            if (depth <= 2) & (stat + 300 * depth < alpha):
                score = qsearch(S, ply, alpha, beta)
                if score < alpha:
                    return score

    # internal iterative reduction when no TT move at high depth
    if (ttmv == 0) & (depth >= 5) & pv_node:
        depth -= 1

    moves = S[X_MV][ply]
    scores = S[X_MS][ply]
    n = cb.gen_moves(P, moves, cb.ALL_MOVES)
    score_moves(P, moves, scores, n, ttmv, killers, history, ply)

    best = -INF
    best_move = np.int64(0)
    legal = 0
    flag = FLAG_UPPER
    futile = (not pv_node) & (not incheck) & (depth <= 5) & (stat + 120 * depth + 60 <= alpha)
    side = P[cb.SIDE]
    for i in range(n):
        m = pick_next(moves, scores, n, i)
        flags = cb.mv_flags(m)
        is_quiet = ((flags & cb.F_CAPTURE) == 0) & (cb.mv_promo(m) == 0)
        # late move pruning / futility pruning of quiet moves
        if is_quiet & (legal > 0) & (not incheck) & (best > -MATE_BOUND):
            if futile:
                continue
            if (not pv_node) & (depth <= 3) & (legal >= 4 + 3 * depth * depth):
                continue
        if not do_move(P, undo, ply, m, acc, net, ctl):
            cb.unmake_move(P, undo, ply)
            continue
        legal += 1
        keys[ply + 1] = P[cb.HASH]
        gives_check = cb.in_check(P)
        new_depth = depth - 1
        # PVS with late move reductions: the first move gets the full window at full depth;
        # later moves get a zero window, possibly reduced, and are re-searched at full depth
        # and then with the full window when they beat alpha.
        r = 0
        lo = -beta
        if legal > 1:
            lo = -alpha - 1
            if is_quiet & (depth >= 3) & (legal > 3) & (not incheck) & (not gives_check):
                r = LMR_TABLE[min(depth, MAX_DEPTH), min(legal, 63)]
                if pv_node:
                    r -= 1
                if scores[i] >= 800_000:
                    r -= 1
                if r < 0:
                    r = 0
                if r > new_depth - 1:
                    r = max(0, new_depth - 1)
        d = new_depth - r
        score = 0
        while True:
            score = -negamax(S, ply + 1, d, lo, -alpha)
            if (legal == 1) | (score <= alpha) | (ctl[C_ABORT] != 0):
                break
            if d < new_depth:
                d = new_depth  # the reduced search beat alpha: full depth, zero window
            elif (lo == -alpha - 1) & (score < beta):
                lo = -beta  # full window
            else:
                break
        cb.unmake_move(P, undo, ply)
        if ctl[C_ABORT] != 0:
            return 0
        if root:
            ctl[C_ROOT_MOVES_DONE] = legal
        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                flag = FLAG_EXACT
                if root:
                    ctl[C_ROOT_MOVE] = m
                    ctl[C_ROOT_SCORE] = score
                if alpha >= beta:
                    flag = FLAG_LOWER
                    if is_quiet:
                        if killers[ply, 0] != m:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = m
                        history[side, cb.mv_from(m), cb.mv_to(m)] += depth * depth
                        if history[side, cb.mv_from(m), cb.mv_to(m)] > 500_000:
                            for a in range(64):
                                for b in range(64):
                                    history[side, a, b] //= 2
                    break
    if legal == 0:
        return -MATE + ply if incheck else 0
    tt_store(tt_key, tt_val, key, best_move, score_to_tt(best, ply), depth, flag)
    return best


@jit
def search_root(S, depth, alpha, beta):  # type: ignore[no-untyped-def]
    P = S[X_P]
    ctl = S[X_CTL]
    S[X_KEYS][0] = P[cb.HASH]
    ctl[C_ROOT_MOVES_DONE] = 0
    ctl[C_NULL_PLY] = 0
    return negamax(S, np.int64(0), depth, alpha, beta)


@jit
def extract_pv(P, undo, mv, tt_key, tt_val, out, max_len):  # type: ignore[no-untyped-def]
    """Follow TT moves from the root; returns the pv length.  Restores P."""
    n = 0
    while n < max_len:
        val = tt_probe(tt_key, tt_val, P[cb.HASH])
        if val == 0:
            break
        m = tt_move(val)
        if m == 0:
            break
        # verify the move is legal here
        cnt = cb.gen_moves(P, mv[n], cb.ALL_MOVES)
        found = False
        for i in range(cnt):
            if mv[n, i] == m:
                found = True
        if not found:
            break
        if not cb.make_move(P, undo, n, m):
            cb.unmake_move(P, undo, n)
            break
        out[n] = m
        n += 1
    for i in range(n - 1, -1, -1):
        cb.unmake_move(P, undo, i)
    return n


# ----------------------------------------------------------------------------- driver


class Searcher:
    """Owns the arrays; one instance per game so the TT and history survive between moves."""

    def __init__(
        self,
        net: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        use_nnue: bool = True,
    ) -> None:
        self.P = cb.new_position()
        self.undo = cb.new_undo()
        self.mv = cb.new_movelists()
        self.ms = cb.new_movelists()
        if net is None:
            net = nnue.random_net(16)
            use_nnue = False
        self.net = net
        self.hidden = int(net[0].shape[1])
        self.acc = nnue.new_acc(self.hidden)
        self.tt_key = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_val = np.zeros(TT_SIZE, dtype=np.int64)
        self.killers = np.zeros((cb.MAX_PLY + 2, 2), dtype=np.int64)
        self.history = np.zeros((2, 64, 64), dtype=np.int64)
        self.keys = np.zeros(cb.MAX_PLY + 2, dtype=np.int64)
        self.ghist = np.zeros(1024, dtype=np.int64)
        self.ctl = np.zeros(CTL_SIZE, dtype=np.int64)
        self.ctf = np.zeros(4, dtype=np.float64)
        self.ctl[C_USE_NNUE] = 1 if use_nnue else 0
        self.pv = np.zeros(cb.MAX_PLY, dtype=np.int64)

    def set_position(self, board, history_keys: list[int] | None = None) -> None:  # type: ignore[no-untyped-def]
        cb.from_board(board, self.P)
        nnue.refresh(self.acc, 0, self.P, self.net[0], self.net[1])
        hist = list(history_keys or [])
        hist.append(int(self.P[cb.HASH]))
        hist = hist[-1000:]
        self.ghist[: len(hist)] = hist
        self.ctl[C_GHIST_N] = len(hist)

    def clear(self) -> None:
        self.tt_key[:] = 0
        self.tt_val[:] = 0
        self.killers[:] = 0
        self.history[:] = 0

    def new_search(self) -> None:
        self.killers[:] = 0
        self.history //= 4
        for k in (
            C_NODES,
            C_QNODES,
            C_ABORT,
            C_SELDEPTH,
            C_TT_HITS,
            C_ROOT_MOVE,
            C_ROOT_SCORE,
            C_ROOT_DEPTH,
        ):
            self.ctl[k] = 0
        self.ctl[C_NEXT_CHECK] = NODES_PER_CLOCK_CHECK

    def search(
        self, max_depth: int = MAX_DEPTH, time_budget: float | None = None, node_limit: int = 0
    ) -> tuple[int, int, int, list[int], dict[str, float]]:
        """Iterative deepening.  Returns (best_move, score, depth, pv list, stats dict)."""
        self.new_search()
        start = time.perf_counter()
        self.ctl[C_NODE_LIMIT] = node_limit
        self.ctf[F_DEADLINE] = start + time_budget if time_budget is not None else 0.0
        best_move = 0
        best_score = 0
        completed = 0
        pv_list: list[int] = []
        args = (
            self.P,
            self.undo,
            self.mv,
            self.ms,
            self.acc,
            self.net,
            self.tt_key,
            self.tt_val,
            self.killers,
            self.history,
            self.keys,
            self.ghist,
            self.ctl,
            self.ctf,
        )
        for depth in range(1, max_depth + 1):
            alpha, beta = -INF, INF
            window = 30
            if depth >= 5:
                alpha, beta = best_score - window, best_score + window
            while True:
                self.ctl[C_ROOT_MOVE] = 0
                score = int(search_root(args, depth, alpha, beta))
                if self.ctl[C_ABORT]:
                    break
                if score <= alpha:
                    alpha = max(-INF, alpha - window)
                    window *= 3
                elif score >= beta:
                    beta = min(INF, beta + window)
                    window *= 3
                else:
                    break
            aborted = bool(self.ctl[C_ABORT])
            if aborted:
                # keep an improved root move from the partial iteration only if it was fully
                # searched and raised alpha above the previous best
                if (
                    self.ctl[C_ROOT_MOVE] != 0
                    and self.ctl[C_ROOT_MOVES_DONE] >= 1
                    and self.ctl[C_ROOT_SCORE] > best_score
                ):
                    best_move = int(self.ctl[C_ROOT_MOVE])
                    best_score = int(self.ctl[C_ROOT_SCORE])
                break
            if self.ctl[C_ROOT_MOVE] != 0:
                best_move = int(self.ctl[C_ROOT_MOVE])
            best_score = score
            completed = depth
            n = int(extract_pv(self.P, self.undo, self.mv, self.tt_key, self.tt_val, self.pv, 16))
            pv_list = [int(x) for x in self.pv[:n]]
            if best_move == 0 and pv_list:
                best_move = pv_list[0]
            if abs(best_score) >= MATE_BOUND and depth >= 3:
                break
            if time_budget is not None:
                elapsed = time.perf_counter() - start
                if elapsed > time_budget * 0.55:
                    break
        elapsed = time.perf_counter() - start
        stats = {
            "nodes": int(self.ctl[C_NODES]),
            "qnodes": int(self.ctl[C_QNODES]),
            "depth": completed,
            "seldepth": int(self.ctl[C_SELDEPTH]),
            "tt_hits": int(self.ctl[C_TT_HITS]),
            "elapsed": elapsed,
            "nps": int(self.ctl[C_NODES] / elapsed) if elapsed > 0 else 0,
        }
        return best_move, best_score, completed, pv_list, stats
