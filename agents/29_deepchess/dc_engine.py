"""numba chess core for the DeepChess agent: board, move generation, make/unmake, search.

Written for this agent from scratch (mailbox board, ray walking, pseudo-legal generation
with legality checked after make). The evaluation is the DeepChess network from
``agent.py``; its first layer is kept incrementally in two side-relative accumulators the
way NNUE-style engines do, so a leaf costs the small dense layers only.

Everything numeric lives in numpy arrays so the whole search can be compiled by numba.

Board layout: square index = rank * 8 + file, a1 = 0, h8 = 63.
Piece codes: 0 empty, 1..6 white P N B R Q K, 7..12 black P N B R Q K.
Move encoding (int32): from | to << 6 | promo << 12 | flags << 16
    promo: 0 none, 2 knight, 3 bishop, 4 rook, 5 queen (piece type index + 1)
    flags: 1 capture, 2 en passant, 4 castle, 8 double pawn push
"""

from __future__ import annotations

import numpy as np
from numba import njit, uint64, int64, int32, int8, float32, boolean

# ------------------------------------------------------------------------------ constants

WHITE, BLACK = 0, 1
EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12

F_CAPTURE, F_EP, F_CASTLE, F_DOUBLE = 1, 2, 4, 8

MAX_PLY = 96
MAX_MOVES = 256
MOVE_STACK = MAX_PLY * MAX_MOVES
MAX_HISTORY = 1024  # game history + search path hashes

MATE = 100_000
MATE_BOUND = MATE - 1000
INF = MATE + 1
DRAW = 0

TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

# state vector indices
S_TURN, S_CASTLING, S_EP, S_HALFMOVE, S_PLY, S_HIST_LEN = 0, 1, 2, 3, 4, 5
S_KING_W, S_KING_B, S_ROOT_HIST = 6, 7, 8
STATE_SIZE = 16

# stats vector indices
ST_NODES, ST_QNODES, ST_MAX_NODES, ST_ABORT, ST_TT_HITS, ST_SELDEPTH, ST_TT_STORES = 0, 1, 2, 3, 4, 5, 6
STATS_SIZE = 8

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 20000,
                        100, 320, 330, 500, 900, 20000], dtype=np.int64)

# castling right bits: 1 white K, 2 white Q, 4 black K, 8 black Q
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 15 - 2   # a1 rook
CASTLE_MASK[7] = 15 - 1   # h1 rook
CASTLE_MASK[4] = 15 - 3   # e1 king
CASTLE_MASK[56] = 15 - 8  # a8 rook
CASTLE_MASK[63] = 15 - 4  # h8 rook
CASTLE_MASK[60] = 15 - 12  # e8 king

# ---------------------------------------------------------------------- attack tables

KNIGHT_D = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_D = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
BISHOP_D = ((1, 1), (1, -1), (-1, -1), (-1, 1))
ROOK_D = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _build_jump_table(deltas: tuple) -> np.ndarray:
    table = np.full((64, 9), -1, dtype=np.int64)
    for sq in range(64):
        r, f = divmod(sq, 8)
        n = 0
        for df, dr in deltas:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                table[sq, n] = rr * 8 + ff
                n += 1
    return table


def _build_ray_table(deltas: tuple) -> np.ndarray:
    """table[sq, dir, step] = target square or -1 (rays stop at the edge)."""
    table = np.full((64, len(deltas), 8), -1, dtype=np.int64)
    for sq in range(64):
        r, f = divmod(sq, 8)
        for d, (df, dr) in enumerate(deltas):
            rr, ff = r + dr, f + df
            step = 0
            while 0 <= rr < 8 and 0 <= ff < 8:
                table[sq, d, step] = rr * 8 + ff
                step += 1
                rr += dr
                ff += df
    return table


KNIGHT_T = _build_jump_table(KNIGHT_D)
KING_T = _build_jump_table(KING_D)
BISHOP_RAYS = _build_ray_table(BISHOP_D)
ROOK_RAYS = _build_ray_table(ROOK_D)

# pawn attack sources: PAWN_ATTACKERS[color, sq, k] = squares from which a pawn of `color`
# attacks sq
PAWN_ATTACKERS = np.full((2, 64, 2), -1, dtype=np.int64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    for _color, _dr in ((WHITE, -1), (BLACK, 1)):
        _n = 0
        for _df in (-1, 1):
            _rr, _ff = _r + _dr, _f + _df
            if 0 <= _rr < 8 and 0 <= _ff < 8:
                PAWN_ATTACKERS[_color, _sq, _n] = _rr * 8 + _ff
                _n += 1

# zobrist keys, fixed seed so the hash is stable between runs
_zrng = np.random.default_rng(0x5EED_29)
ZOBRIST = _zrng.integers(0, 2**63 - 1, size=(13, 64), dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
ZOBRIST[0, :] = 0
Z_CASTLE = _zrng.integers(0, 2**63 - 1, size=16, dtype=np.int64).astype(np.uint64)
Z_CASTLE[0] = 0
Z_EP = _zrng.integers(0, 2**63 - 1, size=8, dtype=np.int64).astype(np.uint64)
Z_SIDE = np.uint64(_zrng.integers(0, 2**63 - 1, dtype=np.int64))


# ------------------------------------------------------------------------------- helpers


@njit(int64(int64), cache=False, nogil=True, inline="always")
def piece_color(p):
    return 0 if p <= 6 else 1


@njit(int64(int64), cache=False, nogil=True, inline="always")
def piece_type(p):
    """0 pawn .. 5 king (p must be non-empty)."""
    return (p - 1) % 6


@njit(int32(int64, int64, int64, int64), cache=False, nogil=True, inline="always")
def make_move_code(frm, to, promo, flags):
    return int32(frm | (to << 6) | (promo << 12) | (flags << 16))


@njit(boolean(int8[:], int64, int64, int64[:, :], int64[:, :], int64[:, :, :], int64[:, :, :],
              int64[:, :, :]), cache=False, nogil=True)
def is_attacked(board, sq, by_color, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers):
    """True if `sq` is attacked by any piece of `by_color`."""
    base = 6 * by_color
    # pawns
    for k in range(2):
        s = pawn_attackers[by_color, sq, k]
        if s >= 0 and board[s] == base + 1:
            return True
    # knights
    for k in range(8):
        s = knight_t[sq, k]
        if s < 0:
            break
        if board[s] == base + 2:
            return True
    # king
    for k in range(8):
        s = king_t[sq, k]
        if s < 0:
            break
        if board[s] == base + 6:
            return True
    # bishops / queens
    for d in range(4):
        for step in range(8):
            s = bishop_rays[sq, d, step]
            if s < 0:
                break
            p = board[s]
            if p != 0:
                if p == base + 3 or p == base + 5:
                    return True
                break
    # rooks / queens
    for d in range(4):
        for step in range(8):
            s = rook_rays[sq, d, step]
            if s < 0:
                break
            p = board[s]
            if p != 0:
                if p == base + 4 or p == base + 5:
                    return True
                break
    return False


@njit(int64(int8[:], int64[:], int32[:], int64, boolean, int64[:, :], int64[:, :],
            int64[:, :, :], int64[:, :, :], int64[:, :, :]), cache=False, nogil=True)
def gen_moves(board, state, out, start, captures_only, knight_t, king_t, bishop_rays,
              rook_rays, pawn_attackers):
    """Pseudo-legal moves for the side to move written to out[start:], returns the end."""
    turn = state[S_TURN]
    base = 6 * turn
    opp_lo = 7 if turn == 0 else 1
    opp_hi = 12 if turn == 0 else 6
    n = start
    ep = state[S_EP]
    for frm in range(64):
        p = board[frm]
        if p == 0 or piece_color(p) != turn:
            continue
        pt = p - base  # 1..6
        if pt == 1:
            # pawn
            if turn == 0:
                fwd = frm + 8
                start_rank = 1
                promo_rank = 7
            else:
                fwd = frm - 8
                start_rank = 6
                promo_rank = 0
            rank = frm >> 3
            file = frm & 7
            # captures
            for df in (-1, 1):
                f2 = file + df
                if f2 < 0 or f2 > 7:
                    continue
                to = fwd + df
                q = board[to]
                if q >= opp_lo and q <= opp_hi:
                    if (to >> 3) == promo_rank:
                        for promo in (5, 2, 4, 3):
                            out[n] = make_move_code(frm, to, promo, F_CAPTURE)
                            n += 1
                    else:
                        out[n] = make_move_code(frm, to, 0, F_CAPTURE)
                        n += 1
                elif to == ep:
                    out[n] = make_move_code(frm, to, 0, F_CAPTURE | F_EP)
                    n += 1
            # pushes
            if board[fwd] == 0:
                if (fwd >> 3) == promo_rank:
                    for promo in (5, 2, 4, 3):
                        # promotions count as tactical in quiescence (queen only there)
                        if captures_only and promo != 5:
                            continue
                        out[n] = make_move_code(frm, fwd, promo, 0)
                        n += 1
                elif not captures_only:
                    out[n] = make_move_code(frm, fwd, 0, 0)
                    n += 1
                    if rank == start_rank:
                        fwd2 = fwd + 8 if turn == 0 else fwd - 8
                        if board[fwd2] == 0:
                            out[n] = make_move_code(frm, fwd2, 0, F_DOUBLE)
                            n += 1
        elif pt == 2 or pt == 6:
            table = knight_t if pt == 2 else king_t
            for k in range(8):
                to = table[frm, k]
                if to < 0:
                    break
                q = board[to]
                if q == 0:
                    if not captures_only:
                        out[n] = make_move_code(frm, to, 0, 0)
                        n += 1
                elif q >= opp_lo and q <= opp_hi:
                    out[n] = make_move_code(frm, to, 0, F_CAPTURE)
                    n += 1
            if pt == 6 and not captures_only:
                # castling: rights, empty squares and no attack on king path
                castling = state[S_CASTLING]
                if turn == 0 and frm == 4:
                    if (castling & 1) and board[5] == 0 and board[6] == 0 and board[7] == WR:
                        if (not is_attacked(board, 4, 1, knight_t, king_t, bishop_rays,
                                            rook_rays, pawn_attackers)
                                and not is_attacked(board, 5, 1, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)
                                and not is_attacked(board, 6, 1, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)):
                            out[n] = make_move_code(4, 6, 0, F_CASTLE)
                            n += 1
                    if ((castling & 2) and board[3] == 0 and board[2] == 0 and board[1] == 0
                            and board[0] == WR):
                        if (not is_attacked(board, 4, 1, knight_t, king_t, bishop_rays,
                                            rook_rays, pawn_attackers)
                                and not is_attacked(board, 3, 1, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)
                                and not is_attacked(board, 2, 1, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)):
                            out[n] = make_move_code(4, 2, 0, F_CASTLE)
                            n += 1
                elif turn == 1 and frm == 60:
                    if (castling & 4) and board[61] == 0 and board[62] == 0 and board[63] == BR:
                        if (not is_attacked(board, 60, 0, knight_t, king_t, bishop_rays,
                                            rook_rays, pawn_attackers)
                                and not is_attacked(board, 61, 0, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)
                                and not is_attacked(board, 62, 0, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)):
                            out[n] = make_move_code(60, 62, 0, F_CASTLE)
                            n += 1
                    if ((castling & 8) and board[59] == 0 and board[58] == 0 and board[57] == 0
                            and board[56] == BR):
                        if (not is_attacked(board, 60, 0, knight_t, king_t, bishop_rays,
                                            rook_rays, pawn_attackers)
                                and not is_attacked(board, 59, 0, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)
                                and not is_attacked(board, 58, 0, knight_t, king_t, bishop_rays,
                                                    rook_rays, pawn_attackers)):
                            out[n] = make_move_code(60, 58, 0, F_CASTLE)
                            n += 1
        else:
            # sliders: bishop 3, rook 4, queen 5
            if pt == 3 or pt == 5:
                for d in range(4):
                    for step in range(8):
                        to = bishop_rays[frm, d, step]
                        if to < 0:
                            break
                        q = board[to]
                        if q == 0:
                            if not captures_only:
                                out[n] = make_move_code(frm, to, 0, 0)
                                n += 1
                        else:
                            if q >= opp_lo and q <= opp_hi:
                                out[n] = make_move_code(frm, to, 0, F_CAPTURE)
                                n += 1
                            break
            if pt == 4 or pt == 5:
                for d in range(4):
                    for step in range(8):
                        to = rook_rays[frm, d, step]
                        if to < 0:
                            break
                        q = board[to]
                        if q == 0:
                            if not captures_only:
                                out[n] = make_move_code(frm, to, 0, 0)
                                n += 1
                        else:
                            if q >= opp_lo and q <= opp_hi:
                                out[n] = make_move_code(frm, to, 0, F_CAPTURE)
                                n += 1
                            break
    return n


# ---------------------------------------------------------------- accumulator (NN layer 1)


@njit(int64(int64, int64, int64), cache=False, nogil=True, inline="always")
def feature_index(piece, sq, perspective):
    """DeepChess feature index of `piece` on `sq` seen from `perspective` (0 white)."""
    pt = piece_type(piece)
    ours = 0 if piece_color(piece) == perspective else 1
    flip = 0 if perspective == 0 else 56
    return (pt * 2 + ours) * 64 + (sq ^ flip)


@njit((float32[:, :], float32[:, :], float32[:], int64, int64), cache=False, nogil=True,
      inline="always")
def acc_add(acc, w1, b1, piece, sq):
    n = acc.shape[1]
    i0 = feature_index(piece, sq, 0)
    i1 = feature_index(piece, sq, 1)
    for j in range(n):
        acc[0, j] += w1[i0, j]
        acc[1, j] += w1[i1, j]


@njit((float32[:, :], float32[:, :], float32[:], int64, int64), cache=False, nogil=True,
      inline="always")
def acc_sub(acc, w1, b1, piece, sq):
    n = acc.shape[1]
    i0 = feature_index(piece, sq, 0)
    i1 = feature_index(piece, sq, 1)
    for j in range(n):
        acc[0, j] -= w1[i0, j]
        acc[1, j] -= w1[i1, j]


@njit((int8[:], float32[:, :], float32[:, :], float32[:]), cache=False, nogil=True)
def acc_refresh(board, acc, w1, b1):
    n = acc.shape[1]
    for j in range(n):
        acc[0, j] = b1[j]
        acc[1, j] = b1[j]
    for sq in range(64):
        p = board[sq]
        if p != 0:
            acc_add(acc, w1, b1, p, sq)


# ------------------------------------------------------------------------- make / unmake

# undo record: move, captured piece, castling, ep, halfmove, hash (as int64 bits)
U_MOVE, U_CAPT, U_CASTLING, U_EP, U_HALF, U_HASH = 0, 1, 2, 3, 4, 5
UNDO_SIZE = 6


@njit((int8[:], int64[:], uint64[:], int32, int64[:, :], float32[:, :, :], float32[:, :],
       float32[:], uint64[:, :], uint64[:], uint64[:], uint64, int64[:], uint64[:]),
      cache=False, nogil=True)
def make_move(board, state, hash_arr, move, undo, acc_stack, w1, b1, zobrist, z_castle,
              z_ep, z_side, castle_mask, hist):
    ply = state[S_PLY]
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 15
    flags = (move >> 16) & 15
    turn = state[S_TURN]
    piece = board[frm]
    captured = board[to]
    h = hash_arr[0]

    undo[ply, U_MOVE] = move
    undo[ply, U_CASTLING] = state[S_CASTLING]
    undo[ply, U_EP] = state[S_EP]
    undo[ply, U_HALF] = state[S_HALFMOVE]
    undo[ply, U_HASH] = int64(h)

    # copy accumulator to the next ply and update incrementally
    acc = acc_stack[ply + 1]
    prev = acc_stack[ply]
    n = acc.shape[1]
    for j in range(n):
        acc[0, j] = prev[0, j]
        acc[1, j] = prev[1, j]

    # en passant file hashing out
    if state[S_EP] >= 0:
        h ^= z_ep[state[S_EP] & 7]

    if flags & F_EP:
        cap_sq = to - 8 if turn == 0 else to + 8
        captured = board[cap_sq]
        board[cap_sq] = 0
        h ^= zobrist[captured, cap_sq]
        acc_sub(acc, w1, b1, captured, cap_sq)
    elif captured != 0:
        h ^= zobrist[captured, to]
        acc_sub(acc, w1, b1, captured, to)
    undo[ply, U_CAPT] = captured

    # move the piece
    board[frm] = 0
    h ^= zobrist[piece, frm]
    acc_sub(acc, w1, b1, piece, frm)
    placed = piece
    if promo != 0:
        placed = promo + 6 * turn
    board[to] = int8(placed)
    h ^= zobrist[placed, to]
    acc_add(acc, w1, b1, placed, to)

    if flags & F_CASTLE:
        if to == 6:
            rf, rt = 7, 5
        elif to == 2:
            rf, rt = 0, 3
        elif to == 62:
            rf, rt = 63, 61
        else:
            rf, rt = 56, 59
        rook = board[rf]
        board[rf] = 0
        board[rt] = rook
        h ^= zobrist[rook, rf]
        h ^= zobrist[rook, rt]
        acc_sub(acc, w1, b1, rook, rf)
        acc_add(acc, w1, b1, rook, rt)

    # king square
    if piece == WK:
        state[S_KING_W] = to
    elif piece == BK:
        state[S_KING_B] = to

    # castling rights
    old_c = state[S_CASTLING]
    new_c = old_c & castle_mask[frm] & castle_mask[to]
    if new_c != old_c:
        h ^= z_castle[old_c]
        h ^= z_castle[new_c]
        state[S_CASTLING] = new_c

    # en passant square
    if flags & F_DOUBLE:
        state[S_EP] = (frm + to) >> 1
        h ^= z_ep[frm & 7]
    else:
        state[S_EP] = -1

    # halfmove clock
    if piece_type(piece) == 0 or captured != 0:
        state[S_HALFMOVE] = 0
    else:
        state[S_HALFMOVE] += 1

    h ^= z_side
    state[S_TURN] = 1 - turn
    state[S_PLY] = ply + 1
    hash_arr[0] = h
    hist[state[S_HIST_LEN]] = h
    state[S_HIST_LEN] += 1


@njit((int8[:], int64[:], uint64[:], int64[:, :]), cache=False, nogil=True)
def unmake_move(board, state, hash_arr, undo):
    ply = state[S_PLY] - 1
    move = undo[ply, U_MOVE]
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 15
    flags = (move >> 16) & 15
    turn = 1 - state[S_TURN]  # side that moved
    piece = board[to]
    if promo != 0:
        piece = 1 + 6 * turn  # back to a pawn
    board[frm] = int8(piece)
    board[to] = 0
    captured = undo[ply, U_CAPT]
    if flags & F_EP:
        cap_sq = to - 8 if turn == 0 else to + 8
        board[cap_sq] = int8(captured)
    elif captured != 0:
        board[to] = int8(captured)
    if flags & F_CASTLE:
        if to == 6:
            rf, rt = 7, 5
        elif to == 2:
            rf, rt = 0, 3
        elif to == 62:
            rf, rt = 63, 61
        else:
            rf, rt = 56, 59
        board[rf] = board[rt]
        board[rt] = 0
    if piece == WK:
        state[S_KING_W] = frm
    elif piece == BK:
        state[S_KING_B] = frm
    state[S_CASTLING] = undo[ply, U_CASTLING]
    state[S_EP] = undo[ply, U_EP]
    state[S_HALFMOVE] = undo[ply, U_HALF]
    hash_arr[0] = uint64(undo[ply, U_HASH])
    state[S_TURN] = turn
    state[S_PLY] = ply
    state[S_HIST_LEN] -= 1


@njit(boolean(int8[:], int64[:], int64[:, :], int64[:, :], int64[:, :, :], int64[:, :, :],
              int64[:, :, :]), cache=False, nogil=True, inline="always")
def in_check(board, state, knight_t, king_t, bishop_rays, rook_rays, pawn_attackers):
    turn = state[S_TURN]
    ksq = state[S_KING_W] if turn == 0 else state[S_KING_B]
    return is_attacked(board, ksq, 1 - turn, knight_t, king_t, bishop_rays, rook_rays,
                       pawn_attackers)


@njit(boolean(int8[:], int64[:], int64[:, :], int64[:, :], int64[:, :, :], int64[:, :, :],
              int64[:, :, :]), cache=False, nogil=True, inline="always")
def left_king_in_check(board, state, knight_t, king_t, bishop_rays, rook_rays,
                       pawn_attackers):
    """After make_move: is the side that just moved leaving its king attacked?"""
    mover = 1 - state[S_TURN]
    ksq = state[S_KING_W] if mover == 0 else state[S_KING_B]
    return is_attacked(board, ksq, state[S_TURN], knight_t, king_t, bishop_rays, rook_rays,
                       pawn_attackers)


# ------------------------------------------------------------------------------- perft


@njit(int64(int8[:], int64[:], uint64[:], int64[:, :], int32[:], float32[:, :, :],
            float32[:, :], float32[:], uint64[:, :], uint64[:], uint64[:], uint64, int64[:],
            uint64[:], int64, int64[:, :], int64[:, :], int64[:, :, :], int64[:, :, :],
            int64[:, :, :]), cache=False, nogil=True)
def perft(board, state, hash_arr, undo, moves, acc_stack, w1, b1, zobrist, z_castle, z_ep,
          z_side, castle_mask, hist, depth, knight_t, king_t, bishop_rays, rook_rays,
          pawn_attackers):
    if depth == 0:
        return 1
    start = state[S_PLY] * MAX_MOVES
    end = gen_moves(board, state, moves, start, False, knight_t, king_t, bishop_rays,
                    rook_rays, pawn_attackers)
    total = 0
    for i in range(start, end):
        m = moves[i]
        make_move(board, state, hash_arr, m, undo, acc_stack, w1, b1, zobrist, z_castle, z_ep,
                  z_side, castle_mask, hist)
        if not left_king_in_check(board, state, knight_t, king_t, bishop_rays, rook_rays,
                                  pawn_attackers):
            total += perft(board, state, hash_arr, undo, moves, acc_stack, w1, b1, zobrist,
                           z_castle, z_ep, z_side, castle_mask, hist, depth - 1, knight_t,
                           king_t, bishop_rays, rook_rays, pawn_attackers)
        unmake_move(board, state, hash_arr, undo)
    return total


# ------------------------------------------------------------------------ python glue


class Position:
    """Arrays for one search; built from a python-chess board."""

    def __init__(self, hidden: int = 256) -> None:
        self.board = np.zeros(64, dtype=np.int8)
        self.state = np.zeros(STATE_SIZE, dtype=np.int64)
        self.hash = np.zeros(1, dtype=np.uint64)
        self.undo = np.zeros((MAX_PLY + 2, UNDO_SIZE), dtype=np.int64)
        self.moves = np.zeros(MOVE_STACK, dtype=np.int32)
        self.acc = np.zeros((MAX_PLY + 2, 2, hidden), dtype=np.float32)
        self.hist = np.zeros(MAX_HISTORY, dtype=np.uint64)

    def set_board(self, b, w1: np.ndarray, b1: np.ndarray,
                  game_hashes: list[int] | None = None) -> None:
        import chess

        self.board[:] = 0
        for sq, piece in b.piece_map().items():
            self.board[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        st = self.state
        st[:] = 0
        st[S_TURN] = 0 if b.turn == chess.WHITE else 1
        c = 0
        if b.has_kingside_castling_rights(chess.WHITE):
            c |= 1
        if b.has_queenside_castling_rights(chess.WHITE):
            c |= 2
        if b.has_kingside_castling_rights(chess.BLACK):
            c |= 4
        if b.has_queenside_castling_rights(chess.BLACK):
            c |= 8
        st[S_CASTLING] = c
        st[S_EP] = b.ep_square if b.ep_square is not None else -1
        st[S_HALFMOVE] = b.halfmove_clock
        st[S_PLY] = 0
        st[S_KING_W] = b.king(chess.WHITE)
        st[S_KING_B] = b.king(chess.BLACK)
        self.hash[0] = compute_hash(self.board, st)
        acc_refresh(self.board, self.acc[0], w1, b1)
        # game history hashes (positions seen this game), then the root
        n = 0
        if game_hashes:
            for h in game_hashes[-(MAX_HISTORY - MAX_PLY - 4):]:
                self.hist[n] = np.uint64(h)
                n += 1
        self.hist[n] = self.hash[0]
        n += 1
        st[S_HIST_LEN] = n
        st[S_ROOT_HIST] = n


def compute_hash(board: np.ndarray, state: np.ndarray) -> np.uint64:
    h = np.uint64(0)
    for sq in range(64):
        p = int(board[sq])
        if p:
            h ^= ZOBRIST[p, sq]
    h ^= Z_CASTLE[int(state[S_CASTLING])]
    if state[S_EP] >= 0:
        h ^= Z_EP[int(state[S_EP]) & 7]
    if state[S_TURN] == 1:
        h ^= Z_SIDE
    return h


def hash_of_board(b) -> int:
    """Zobrist hash of a python-chess board, matching the engine's incremental hash."""
    pos = Position(hidden=1)
    import chess

    pos.board[:] = 0
    for sq, piece in b.piece_map().items():
        pos.board[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    st = pos.state
    st[S_TURN] = 0 if b.turn == chess.WHITE else 1
    c = 0
    if b.has_kingside_castling_rights(chess.WHITE):
        c |= 1
    if b.has_queenside_castling_rights(chess.WHITE):
        c |= 2
    if b.has_kingside_castling_rights(chess.BLACK):
        c |= 4
    if b.has_queenside_castling_rights(chess.BLACK):
        c |= 8
    st[S_CASTLING] = c
    st[S_EP] = b.ep_square if b.ep_square is not None else -1
    return int(compute_hash(pos.board, st))


def move_to_uci(move: int) -> str:
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 15
    s = "abcdefgh"[frm & 7] + str((frm >> 3) + 1) + "abcdefgh"[to & 7] + str((to >> 3) + 1)
    if promo:
        s += "  nbrq"[promo]
    return s
