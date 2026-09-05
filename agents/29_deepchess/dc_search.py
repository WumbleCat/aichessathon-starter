"""numba alpha-beta search on top of ``dc_engine`` for the DeepChess agent.

The whole tree search runs compiled: negamax with alpha-beta and principal variation
search, a transposition table, killer and history move ordering, null-move pruning, late
move reductions, reverse futility and futility pruning, check extension and a quiescence
search with delta pruning. The leaf evaluation is the DeepChess network whose first layer
is kept incrementally in the accumulator that ``dc_engine.make_move`` maintains; the
handcrafted material + piece-square evaluation is available as well, alone or blended.

Time control: the search cannot read the clock from compiled code, so the Python driver
arms a ``threading.Timer`` that writes into the ``stop`` array at the deadline (the search
runs with the GIL released) and additionally caps the node count. Every node checks both.
"""

from __future__ import annotations

import numpy as np
from numba import njit, uint64, int64, int32, int8, float32, boolean

from dc_engine import (
    BISHOP_RAYS, CASTLE_MASK, DRAW, F_CAPTURE, F_CASTLE, F_EP, INF, KING_T, KNIGHT_T, MATE,
    MATE_BOUND, MAX_MOVES, MAX_PLY, PAWN_ATTACKERS, PIECE_VALUE, ROOK_RAYS, S_CASTLING, S_EP,
    S_HALFMOVE, S_HIST_LEN, S_PLY, S_ROOT_HIST, S_TURN, U_CASTLING, U_EP, U_HALF, U_HASH,
    U_MOVE, WK, ZOBRIST, Z_CASTLE, Z_EP, Z_SIDE, gen_moves, in_check, is_attacked,
    left_king_in_check, make_move, piece_color, piece_type, unmake_move,
)


import os as _os
import time as _time

_TIMING = bool(_os.environ.get("DEEPCHESS_COMPILE_TIMING"))
_t_last = _time.process_time()


def _mark(name: str) -> None:
    """Print the CPU seconds spent compiling since the previous mark (opt-in)."""
    global _t_last
    if _TIMING:
        now = _time.process_time()
        print(f"compile {name}: {now - _t_last:.1f}s cpu", flush=True)
        _t_last = now

# ------------------------------------------------------------------------------ constants

TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
SCORE_OFFSET = 1 << 17
TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

# stats vector
ST_NODES, ST_QNODES, ST_MAX_NODES, ST_ABORT, ST_TT_HITS, ST_SELDEPTH = 0, 1, 2, 3, 4, 5
STATS_SIZE = 8

# params vector
P_MODE, P_BLEND = 0, 1  # mode 0 net, 1 hand, 2 blend; blend = network weight in percent
PARAMS_SIZE = 4

WORK_SIZE = 256 + 32 + 32
MAX_GAME_HASHES = 64


# ------------------------------------------------------------------------- repetition


@njit(int64(int64[:], uint64[:]), cache=False, nogil=True)
def is_repetition(state, hist):
    """1 if the current position occurred before in the search path or game history.

    ``hist`` holds up to MAX_GAME_HASHES positions of this game (the ones we were asked to
    move in, two plies apart), then the root, then the search path. The hash includes the
    side to move, so scanning every entry is safe; the halfmove clock bounds the reach.
    """
    n = state[S_HIST_LEN]
    h = hist[n - 1]
    root = state[S_ROOT_HIST] - 1  # index of the root entry
    half = state[S_HALFMOVE]
    # search path (root included): one entry per ply
    lo = n - 1 - half
    i = n - 3
    while i >= root and i >= lo:
        if hist[i] == h:
            return 1
        i -= 2
    if lo >= root:
        return 0
    # game part: entries are two plies apart
    reach = (half - (n - 1 - root)) // 2
    i = root - 1
    lo = root - reach
    while i >= 0 and i >= lo:
        if hist[i] == h:
            return 1
        i -= 1
    return 0


_mark("is_repetition")

# ------------------------------------------------------------------------- null move


@njit((int8[:], int64[:], uint64[:], int64[:, :], float32[:, :, :], uint64[:], uint64[:]),
      cache=False, nogil=True)
def make_null(board, state, hash_arr, undo, acc_stack, hist, z_ep):
    ply = state[S_PLY]
    undo[ply, U_MOVE] = 0
    undo[ply, U_CASTLING] = state[S_CASTLING]
    undo[ply, U_EP] = state[S_EP]
    undo[ply, U_HALF] = state[S_HALFMOVE]
    undo[ply, U_HASH] = int64(hash_arr[0])
    acc = acc_stack[ply + 1]
    prev = acc_stack[ply]
    n = acc.shape[1]
    for j in range(n):
        acc[0, j] = prev[0, j]
        acc[1, j] = prev[1, j]
    h = hash_arr[0]
    if state[S_EP] >= 0:
        h ^= z_ep[state[S_EP] & 7]
    h ^= Z_SIDE
    state[S_EP] = -1
    state[S_TURN] = 1 - state[S_TURN]
    state[S_HALFMOVE] += 1
    state[S_PLY] = ply + 1
    hash_arr[0] = h
    hist[state[S_HIST_LEN]] = h
    state[S_HIST_LEN] += 1


_mark("make_null")

@njit((int64[:], uint64[:], int64[:, :]), cache=False, nogil=True)
def unmake_null(state, hash_arr, undo):
    ply = state[S_PLY] - 1
    state[S_CASTLING] = undo[ply, U_CASTLING]
    state[S_EP] = undo[ply, U_EP]
    state[S_HALFMOVE] = undo[ply, U_HALF]
    hash_arr[0] = uint64(undo[ply, U_HASH])
    state[S_TURN] = 1 - state[S_TURN]
    state[S_PLY] = ply
    state[S_HIST_LEN] -= 1


_mark("unmake_null")

# ------------------------------------------------------------------------ evaluation


@njit(int64(int8[:], int64, int32[:, :]), cache=False, nogil=True)
def hand_eval(board, turn, pst):
    """Material + piece-square evaluation, side to move perspective (mirrors agent.py)."""
    phase = 0
    score_mg = 0
    score_eg = 0
    for sq in range(64):
        p = board[sq]
        if p == 0:
            continue
        white = p <= 6
        pt = piece_type(p)
        if pt == 0:
            val = 100
        elif pt == 1:
            val = 320
            phase += 1
        elif pt == 2:
            val = 330
            phase += 1
        elif pt == 3:
            val = 500
            phase += 2
        elif pt == 4:
            val = 900
            phase += 4
        else:
            val = 0
        psq = sq if white else (sq ^ 56)
        if pt == 5:
            mg = val + pst[5, psq]
            eg = val + pst[6, psq]
        else:
            mg = val + pst[pt, psq]
            eg = mg
        if white:
            score_mg += mg
            score_eg += eg
        else:
            score_mg -= mg
            score_eg -= eg
    if phase > 24:
        phase = 24
    score = (score_mg * phase + score_eg * (24 - phase)) // 24
    score += 10 if turn == 0 else -10
    return score if turn == 0 else -score


_mark("hand_eval")

@njit(int64(int8[:], int64[:], float32[:, :], float32[:, :], float32[:], float32[:, :],
            float32[:], float32[:, :], float32[:], float32[:], float32[:], float32[:],
            int32[:, :], int64[:]), cache=False, nogil=True)
def evaluate_pos(board, state, acc, w1, b1, w2, b2, w3, b3, w4, b4, work, pst, params):
    """Static evaluation in centipawns from the side to move's perspective."""
    mode = params[P_MODE]
    turn = state[S_TURN]
    net = 0
    if mode != 1:
        n1 = w1.shape[1]
        for j in range(n1):
            work[j] = acc[turn, j]
        c = state[S_CASTLING]
        if turn == 0:
            ok = c & 1
            oq = (c >> 1) & 1
            tk = (c >> 2) & 1
            tq = (c >> 3) & 1
        else:
            ok = (c >> 2) & 1
            oq = (c >> 3) & 1
            tk = c & 1
            tq = (c >> 1) & 1
        if ok:
            for j in range(n1):
                work[j] += w1[768, j]
        if oq:
            for j in range(n1):
                work[j] += w1[769, j]
        if tk:
            for j in range(n1):
                work[j] += w1[770, j]
        if tq:
            for j in range(n1):
                work[j] += w1[771, j]
        if state[S_EP] >= 0:
            for j in range(n1):
                work[j] += w1[772, j]
        for j in range(n1):
            v = work[j]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            work[j] = v
        n2 = w2.shape[1]
        o2 = 256
        for k in range(n2):
            s = b2[k]
            for j in range(n1):
                s += work[j] * w2[j, k]
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            work[o2 + k] = s
        n3 = w3.shape[1]
        o3 = 256 + 32
        for k in range(n3):
            s = b3[k]
            for j in range(n2):
                s += work[o2 + j] * w3[j, k]
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            work[o3 + k] = s
        out = b4[0]
        for j in range(n3):
            out += work[o3 + j] * w4[j]
        cp = out * 100.0
        if cp > 30000.0:
            cp = 30000.0
        elif cp < -30000.0:
            cp = -30000.0
        net = int(cp)
    if mode == 0:
        return net
    hand = hand_eval(board, turn, pst)
    if mode == 1:
        return hand
    w = params[P_BLEND]
    return (net * w + hand * (100 - w)) // 100


_mark("evaluate_pos")

@njit(boolean(int8[:], int64), cache=False, nogil=True, inline="always")
def has_non_pawn_material(board, turn):
    lo = 2 if turn == 0 else 8
    hi = 5 if turn == 0 else 11
    for sq in range(64):
        p = board[sq]
        if p >= lo and p <= hi:
            return True
    return False


_mark("has_non_pawn_material")

@njit(int64(int8[:], int32), cache=False, nogil=True, inline="always")
def mvv_lva(board, move):
    """Capture ordering score: victim value first, cheapest attacker first."""
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 15
    flags = (move >> 16) & 15
    victim = 0
    if flags & F_EP:
        victim = 100
    elif flags & F_CAPTURE:
        victim = PIECE_VALUE[board[to]]
    attacker = piece_type(board[frm]) + 1
    score = victim * 10 - attacker
    if promo != 0:
        score += PIECE_VALUE[promo]
    return score


_mark("mvv_lva")

# ------------------------------------------------------------------------- quiescence


@njit(int64(int8[:], int64[:], uint64[:], int64[:, :], int32[:], int64[:], float32[:, :, :],
            float32[:, :], float32[:], float32[:, :], float32[:], float32[:, :], float32[:],
            float32[:], float32[:], uint64[:], float32[:], int32[:, :], int64[:], int64[:], int64[:], int64[:, :], int64[:, :], int64[:, :, :], int64[:, :, :], int64[:, :, :], uint64[:, :], uint64[:], uint64[:], int64[:], int64, int64, int64),
      cache=False, nogil=True)
def quiesce(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1, w2, b2, w3, b3,
            w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask, alpha, beta, ply):
    stats[ST_NODES] += 1
    stats[ST_QNODES] += 1
    if stop[0] != 0 or stats[ST_NODES] >= stats[ST_MAX_NODES]:
        stats[ST_ABORT] = 1
        return 0
    if ply > stats[ST_SELDEPTH]:
        stats[ST_SELDEPTH] = ply
    if ply >= MAX_PLY - 2:
        return evaluate_pos(board, state, acc_stack[state[S_PLY]], w1, b1, w2, b2, w3, b3, w4,
                            b4, work, pst, params)
    checked = in_check(board, state, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers)
    start = state[S_PLY] * MAX_MOVES
    if checked:
        # evasions: every move, no stand pat
        end = gen_moves(board, state, moves, start, False, knight_t, king_t, bishop_rays,
                        rook_rays, pawn_attackers)
        stand = -INF
        best = -INF
        legal = 0
    else:
        stand = evaluate_pos(board, state, acc_stack[state[S_PLY]], w1, b1, w2, b2, w3, b3,
                             w4, b4, work, pst, params)
        if stand >= beta:
            return stand
        if ply >= MAX_PLY - 2:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        end = gen_moves(board, state, moves, start, True, knight_t, king_t, bishop_rays,
                        rook_rays, pawn_attackers)
        legal = 1  # not used for mate detection when not in check
    for i in range(start, end):
        mscore[i] = mvv_lva(board, moves[i])
    for i in range(start, end):
        # selection sort: bring the best remaining move to position i
        bi = i
        bs = mscore[i]
        for j in range(i + 1, end):
            if mscore[j] > bs:
                bs = mscore[j]
                bi = j
        if bi != i:
            tm = moves[i]
            moves[i] = moves[bi]
            moves[bi] = tm
            mscore[bi] = mscore[i]
            mscore[i] = bs
        m = moves[i]
        if not checked:
            # delta pruning
            flags = (m >> 16) & 15
            to = (m >> 6) & 63
            gain = 100 if (flags & F_EP) else PIECE_VALUE[board[to]]
            if ((m >> 12) & 15) != 0:
                gain += 800
            if stand + gain + 200 <= alpha:
                continue
        make_move(board, state, hash_arr, m, undo, acc_stack, w1, b1, zobrist, z_castle, z_ep,
                  Z_SIDE, castle_mask, hist)
        if left_king_in_check(board, state, knight_t, king_t, bishop_rays, rook_rays,
                              pawn_attackers):
            unmake_move(board, state, hash_arr, undo)
            continue
        if checked:
            legal += 1
        score = -quiesce(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1, w2,
                         b2, w3, b3, w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask, -beta,
                         -alpha, ply + 1)
        unmake_move(board, state, hash_arr, undo)
        if stats[ST_ABORT] != 0:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if checked and legal == 0:
        return -MATE + ply
    return best


_mark("quiesce")

# ------------------------------------------------------------------------ main search


@njit(int64(int8[:], int64[:], uint64[:], int64[:, :], int32[:], int64[:], float32[:, :, :],
            float32[:, :], float32[:], float32[:, :], float32[:], float32[:, :], float32[:],
            float32[:], float32[:], uint64[:], float32[:], int32[:, :], int64[:], int64[:], int64[:], int64[:, :], int64[:, :], int64[:, :, :], int64[:, :, :], int64[:, :, :], uint64[:, :], uint64[:], uint64[:], int64[:], uint64[:], int64[:], int32[:, :], int64[:, :, :], int64, int64, int64,
            int64, boolean),
      cache=False, nogil=True)
def search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1, w2, b2, w3, b3,
           w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask, tt_key, tt_data, killers, history,
           depth, alpha, beta, ply, null_ok):
    stats[ST_NODES] += 1
    if stop[0] != 0 or stats[ST_NODES] >= stats[ST_MAX_NODES]:
        stats[ST_ABORT] = 1
        return 0
    if ply >= MAX_PLY - 2:
        return evaluate_pos(board, state, acc_stack[state[S_PLY]], w1, b1, w2, b2, w3, b3, w4,
                            b4, work, pst, params)
    if ply > 0:
        if state[S_HALFMOVE] >= 100 or is_repetition(state, hist) != 0:
            return DRAW
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    checked = in_check(board, state, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers)
    if checked:
        depth += 1
    if depth <= 0:
        return quiesce(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1, w2, b2,
                       w3, b3, w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask, alpha, beta, ply)

    key = hash_arr[0]
    idx = int64(key & uint64(TT_MASK))
    tt_move = int32(0)
    if tt_key[idx] == key:
        data = tt_data[idx]
        tt_move = int32(data & 0xFFFFF)
        tt_depth = (data >> 20) & 0xFF
        tt_flag = (data >> 28) & 3
        tt_score = ((data >> 30) & 0x3FFFF) - SCORE_OFFSET
        if tt_depth >= depth and ply > 0:
            if tt_score > MATE_BOUND:
                tt_score -= ply
            elif tt_score < -MATE_BOUND:
                tt_score += ply
            if tt_flag == TT_EXACT:
                stats[ST_TT_HITS] += 1
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                stats[ST_TT_HITS] += 1
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                stats[ST_TT_HITS] += 1
                return tt_score

    pv_node = beta - alpha > 1
    turn = state[S_TURN]
    static_eval = -INF
    if not checked and (depth <= 2 or (not pv_node and ply > 0)):
        static_eval = evaluate_pos(board, state, acc_stack[state[S_PLY]], w1, b1, w2, b2, w3,
                                   b3, w4, b4, work, pst, params)
    if not checked and not pv_node and ply > 0:
        # reverse futility pruning
        if depth <= 3 and static_eval - 120 * depth >= beta and abs(beta) < MATE_BOUND:
            return static_eval
        # null move pruning
        if null_ok and depth >= 2 and static_eval >= beta and has_non_pawn_material(board, turn):
            r = 3 if depth >= 6 else 2
            make_null(board, state, hash_arr, undo, acc_stack, hist, z_ep)
            score = -search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1,
                            w2, b2, w3, b3, w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask,
                            tt_key, tt_data, killers, history, depth - 1 - r, -beta,
                            -beta + 1, ply + 1, False)
            unmake_null(state, hash_arr, undo)
            if stats[ST_ABORT] != 0:
                return 0
            if score >= beta and abs(score) < MATE_BOUND:
                return score

    start = state[S_PLY] * MAX_MOVES
    end = gen_moves(board, state, moves, start, False, knight_t, king_t, bishop_rays,
                    rook_rays, pawn_attackers)
    k0 = killers[ply, 0]
    k1 = killers[ply, 1]
    for i in range(start, end):
        m = moves[i]
        if m == tt_move:
            mscore[i] = 10_000_000
        elif ((m >> 16) & F_CAPTURE) != 0 or ((m >> 12) & 15) != 0:
            mscore[i] = 1_000_000 + mvv_lva(board, m)
        elif m == k0:
            mscore[i] = 900_000
        elif m == k1:
            mscore[i] = 800_000
        else:
            mscore[i] = history[turn, m & 63, (m >> 6) & 63]

    best = -INF
    best_move = int32(0)
    alpha_orig = alpha
    legal = 0
    for i in range(start, end):
        bi = i
        bs = mscore[i]
        for j in range(i + 1, end):
            if mscore[j] > bs:
                bs = mscore[j]
                bi = j
        if bi != i:
            tm = moves[i]
            moves[i] = moves[bi]
            moves[bi] = tm
            mscore[bi] = mscore[i]
            mscore[i] = bs
        m = moves[i]
        quiet = ((m >> 16) & F_CAPTURE) == 0 and ((m >> 12) & 15) == 0
        make_move(board, state, hash_arr, m, undo, acc_stack, w1, b1, zobrist, z_castle, z_ep,
                  Z_SIDE, castle_mask, hist)
        if left_king_in_check(board, state, knight_t, king_t, bishop_rays, rook_rays,
                              pawn_attackers):
            unmake_move(board, state, hash_arr, undo)
            continue
        if quiet and in_check(board, state, knight_t, king_t, bishop_rays, rook_rays,
                              pawn_attackers):
            quiet = False  # checking moves are never reduced or pruned
        # futility pruning at frontier nodes
        if (quiet and not checked and depth <= 2 and legal > 0 and abs(alpha) < MATE_BOUND
                and static_eval + 150 * depth <= alpha):
            unmake_move(board, state, hash_arr, undo)
            continue
        new_depth = depth - 1
        if legal >= 3 and depth >= 3 and quiet and not checked and m != k0 and m != k1:
            reduction = 1 if legal < 8 else 2
            score = -search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1,
                            w2, b2, w3, b3, w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask,
                            tt_key, tt_data, killers, history, new_depth - reduction,
                            -alpha - 1, -alpha, ply + 1, True)
            if score > alpha and stats[ST_ABORT] == 0:
                score = -search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1,
                                b1, w2, b2, w3, b3, w4, b4, hist, work, pst, params, stats,
                                stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask, tt_key, tt_data, killers, history, new_depth, -beta,
                                -alpha, ply + 1, True)
        elif legal == 0:
            score = -search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1,
                            w2, b2, w3, b3, w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask,
                            tt_key, tt_data, killers, history, new_depth, -beta, -alpha,
                            ply + 1, True)
        else:
            score = -search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1, b1,
                            w2, b2, w3, b3, w4, b4, hist, work, pst, params, stats, stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask,
                            tt_key, tt_data, killers, history, new_depth, -alpha - 1, -alpha,
                            ply + 1, True)
            if alpha < score < beta and stats[ST_ABORT] == 0:
                score = -search(board, state, hash_arr, undo, moves, mscore, acc_stack, w1,
                                b1, w2, b2, w3, b3, w4, b4, hist, work, pst, params, stats,
                                stop, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers, zobrist, z_castle, z_ep, castle_mask, tt_key, tt_data, killers, history, new_depth, -beta,
                                -alpha, ply + 1, True)
        unmake_move(board, state, hash_arr, undo)
        if stats[ST_ABORT] != 0:
            return 0
        legal += 1
        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if quiet:
                        if killers[ply, 0] != m:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = m
                        history[turn, m & 63, (m >> 6) & 63] += depth * depth
                    break
    if legal == 0:
        return -MATE + ply if checked else DRAW

    store = best
    if store > MATE_BOUND:
        store += ply
    elif store < -MATE_BOUND:
        store -= ply
    if best <= alpha_orig:
        flag = TT_UPPER
    elif best >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    if tt_key[idx] != key or ((tt_data[idx] >> 20) & 0xFF) <= depth or flag == TT_EXACT:
        tt_key[idx] = key
        tt_data[idx] = (int64(best_move) | (int64(depth) << 20) | (int64(flag) << 28)
                        | (int64(store + SCORE_OFFSET) << 30))
    return best


_mark("search")
