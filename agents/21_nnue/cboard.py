"""Bitboard chess position for numba: make/unmake, pseudo-legal movegen, attack tests.

Everything lives in flat numpy int64 arrays so that njit functions can pass the whole
position around cheaply.  Squares are 0 = a1 .. 63 = h8 (python-chess order).

Piece codes: 0 empty, 1..6 white P N B R Q K, 7..12 black P N B R Q K.

Position array ``P`` (int64, length PSIZE):
    P[0:64]      board, piece code per square
    P[BB+0..12]  bitboard per piece code (index 0 unused)
    P[OCC+0..2]  occupancy: white, black, all
    P[SIDE]      side to move, 0 white 1 black
    P[EP]        en-passant target square or -1
    P[CASTLE]    castling rights bits: 1 WK 2 WQ 4 BK 8 BQ
    P[HALF]      halfmove clock
    P[HASH]      zobrist key
    P[KSQ+0..1]  king squares
    P[LAST_*]    what the last make_move did (for incremental NNUE updates)

Move encoding (int64): from | to << 6 | promo << 12 | flags << 16
    promo: 0 or piece type 2..5 (N B R Q)
    flags: 1 capture, 2 en passant, 4 castle, 8 double pawn push
"""

from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

# ----------------------------------------------------------------------------- constants

WHITE, BLACK = 0, 1
EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB_, BR, BQ, BK = 7, 8, 9, 10, 11, 12

BB = 64
OCC = 77
SIDE = 80
EP = 81
CASTLE = 82
HALF = 83
HASH = 84
KSQ = 85  # 85, 86
LAST_PIECE = 88  # piece that moved (code before promotion)
LAST_FROM = 89
LAST_TO = 90
LAST_CAPTURED = 91  # captured piece code or 0
LAST_CAPSQ = 92  # square the captured piece stood on
LAST_PROMO = 93  # promoted-to piece code or 0
LAST_ROOK_FROM = 94  # castling rook from (-1 if none)
LAST_ROOK_TO = 95
NONPAWN = 96  # non-pawn material count (both sides, pieces only) for null move guard
PSIZE = 100

F_CAPTURE = 1
F_EP = 2
F_CASTLE = 4
F_DOUBLE = 8

MAX_PLY = 128
MAX_MOVES = 256
UNDO_W = 8  # per-ply undo record width: move, captured, ep, castle, half, hash, unused...

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)

FULL = np.uint64(0xFFFFFFFFFFFFFFFF).view(np.int64)

# ----------------------------------------------------------------------------- intrinsics


@intrinsic
def ctz64(typingctx, x):  # type: ignore[no-untyped-def]
    sig = types.int64(types.int64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return builder.cttz(args[0], ir.Constant(ir.IntType(1), 1))

    return sig, codegen


@intrinsic
def clz64(typingctx, x):  # type: ignore[no-untyped-def]
    sig = types.int64(types.int64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return builder.ctlz(args[0], ir.Constant(ir.IntType(1), 1))

    return sig, codegen


@intrinsic
def popcount64(typingctx, x):  # type: ignore[no-untyped-def]
    sig = types.int64(types.int64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return builder.ctpop(args[0])

    return sig, codegen


# ----------------------------------------------------------------------------- tables


def _u(x: int) -> np.int64:
    return np.array([x & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64).view(np.int64)[0]


def _sq_bb(sq: int) -> int:
    return 1 << sq


def _build_tables() -> tuple[np.ndarray, ...]:
    knight = np.zeros(64, dtype=np.int64)
    king = np.zeros(64, dtype=np.int64)
    pawn_att = np.zeros((2, 64), dtype=np.int64)
    rays = np.zeros((8, 64), dtype=np.int64)  # directions: N NE E SE S SW W NW
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for sq in range(64):
        f, r = sq % 8, sq // 8
        for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                knight[sq] |= _u(_sq_bb(nr * 8 + nf))
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    king[sq] |= _u(_sq_bb(nr * 8 + nf))
        for df in (-1, 1):
            nf, nr = f + df, r + 1
            if 0 <= nf < 8 and 0 <= nr < 8:
                pawn_att[0, sq] |= _u(_sq_bb(nr * 8 + nf))
            nf, nr = f + df, r - 1
            if 0 <= nf < 8 and 0 <= nr < 8:
                pawn_att[1, sq] |= _u(_sq_bb(nr * 8 + nf))
        for d, (df, dr) in enumerate(dirs):
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                rays[d, sq] |= _u(_sq_bb(nr * 8 + nf))
                nf += df
                nr += dr
    # castling rights lost when a piece moves from/to these squares
    castle_mask = np.full(64, 15, dtype=np.int64)
    castle_mask[E1] = 15 & ~(CASTLE_WK | CASTLE_WQ)
    castle_mask[A1] = 15 & ~CASTLE_WQ
    castle_mask[H1] = 15 & ~CASTLE_WK
    castle_mask[E8] = 15 & ~(CASTLE_BK | CASTLE_BQ)
    castle_mask[A8] = 15 & ~CASTLE_BQ
    castle_mask[H8] = 15 & ~CASTLE_BK
    # between[a, b]: squares strictly between a and b on a line, else 0
    between = np.zeros((64, 64), dtype=np.int64)
    for a in range(64):
        for d, (df, dr) in enumerate(dirs):
            f, r = a % 8 + df, a // 8 + dr
            acc = 0
            while 0 <= f < 8 and 0 <= r < 8:
                b = r * 8 + f
                between[a, b] = _u(acc)
                acc |= _sq_bb(b)
                f += df
                r += dr
    rng = np.random.default_rng(20260904)
    zob = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(13, 64), dtype=np.int64)
    zob_side = np.int64(rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64))
    zob_castle = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=16, dtype=np.int64)
    zob_ep = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=8, dtype=np.int64)
    zob[0, :] = 0
    zob_castle[0] = 0
    return knight, king, pawn_att, rays, castle_mask, between, zob, np.array([zob_side]), zob_castle, zob_ep


(
    KNIGHT_ATT,
    KING_ATT,
    PAWN_ATT,
    RAYS,
    CASTLE_MASK,
    BETWEEN,
    ZOB,
    ZOB_SIDE_ARR,
    ZOB_CASTLE,
    ZOB_EP,
) = _build_tables()
ZOB_SIDE = int(ZOB_SIDE_ARR[0])

RANK_MASK = np.array([_u(0xFF << (8 * r)) for r in range(8)], dtype=np.int64)
FILE_MASK = np.array([_u(0x0101010101010101 << f) for f in range(8)], dtype=np.int64)
NOT_RANK8_LOW56 = _u(0x00FFFFFFFFFFFFFF)
WK_EMPTY = _u((1 << F1) | (1 << G1))
WQ_EMPTY = _u((1 << B1) | (1 << C1) | (1 << D1))
BK_EMPTY = _u((1 << F8) | (1 << G8))
BQ_EMPTY = _u((1 << B8) | (1 << C8) | (1 << D8))

# ----------------------------------------------------------------------------- bit helpers


@njit(cache=True)
def bit(sq):  # type: ignore[no-untyped-def]
    return np.int64(1) << np.int64(sq)


@njit(cache=True)
def lsb(bb):  # type: ignore[no-untyped-def]
    return ctz64(bb)


@njit(cache=True)
def msb(bb):  # type: ignore[no-untyped-def]
    return 63 - clz64(bb)


@njit(cache=True)
def popcount(bb):  # type: ignore[no-untyped-def]
    return popcount64(bb)


@njit(cache=True)
def shr8(bb):  # type: ignore[no-untyped-def]
    """Logical right shift by 8 of an int64 bitboard."""
    return (bb >> np.int64(8)) & NOT_RANK8_LOW56


@njit(cache=True)
def piece_color(piece):  # type: ignore[no-untyped-def]
    return 1 if piece >= 7 else 0


@njit(cache=True)
def piece_type(piece):  # type: ignore[no-untyped-def]
    return piece - 6 if piece >= 7 else piece


@njit(cache=True)
def make_piece(color, ptype):  # type: ignore[no-untyped-def]
    return ptype + 6 * color


@njit(cache=True)
def mv_from(move):  # type: ignore[no-untyped-def]
    return move & 63


@njit(cache=True)
def mv_to(move):  # type: ignore[no-untyped-def]
    return (move >> 6) & 63


@njit(cache=True)
def mv_promo(move):  # type: ignore[no-untyped-def]
    return (move >> 12) & 7


@njit(cache=True)
def mv_flags(move):  # type: ignore[no-untyped-def]
    return (move >> 16) & 15


@njit(cache=True)
def encode_move(frm, to, promo, flags):  # type: ignore[no-untyped-def]
    return np.int64(frm) | (np.int64(to) << 6) | (np.int64(promo) << 12) | (np.int64(flags) << 16)


# ----------------------------------------------------------------------------- attacks


@njit(cache=True)
def slider_attacks_dir(sq, occ, d):  # type: ignore[no-untyped-def]
    """Attacks from ``sq`` along ray direction ``d`` given occupancy."""
    att = RAYS[d, sq]
    blockers = att & occ
    if blockers != 0:
        if d <= 2 or d == 7:  # N NE E NW: positive directions -> first blocker is lsb
            first = lsb(blockers)
        else:
            first = msb(blockers)
        att &= ~RAYS[d, first]
    return att


@njit(cache=True)
def bishop_attacks(sq, occ):  # type: ignore[no-untyped-def]
    return (
        slider_attacks_dir(sq, occ, 1)
        | slider_attacks_dir(sq, occ, 3)
        | slider_attacks_dir(sq, occ, 5)
        | slider_attacks_dir(sq, occ, 7)
    )


@njit(cache=True)
def rook_attacks(sq, occ):  # type: ignore[no-untyped-def]
    return (
        slider_attacks_dir(sq, occ, 0)
        | slider_attacks_dir(sq, occ, 2)
        | slider_attacks_dir(sq, occ, 4)
        | slider_attacks_dir(sq, occ, 6)
    )


@njit(cache=True)
def attackers_to(P, sq, by, occ):  # type: ignore[no-untyped-def]
    """Bitboard of pieces of colour ``by`` attacking ``sq`` with occupancy ``occ``."""
    base = 6 * by
    att = PAWN_ATT[1 - by, sq] & P[BB + base + PAWN]
    att |= KNIGHT_ATT[sq] & P[BB + base + KNIGHT]
    att |= KING_ATT[sq] & P[BB + base + KING]
    bq = P[BB + base + BISHOP] | P[BB + base + QUEEN]
    if bq != 0:
        att |= bishop_attacks(sq, occ) & bq
    rq = P[BB + base + ROOK] | P[BB + base + QUEEN]
    if rq != 0:
        att |= rook_attacks(sq, occ) & rq
    return att


@njit(cache=True)
def is_attacked(P, sq, by):  # type: ignore[no-untyped-def]
    occ = P[OCC + 2]
    base = 6 * by
    if PAWN_ATT[1 - by, sq] & P[BB + base + PAWN]:
        return True
    if KNIGHT_ATT[sq] & P[BB + base + KNIGHT]:
        return True
    if KING_ATT[sq] & P[BB + base + KING]:
        return True
    bq = P[BB + base + BISHOP] | P[BB + base + QUEEN]
    if bq != 0:
        if (bishop_attacks(sq, occ) & bq) != 0:
            return True
    rq = P[BB + base + ROOK] | P[BB + base + QUEEN]
    if rq != 0:
        if (rook_attacks(sq, occ) & rq) != 0:
            return True
    return False


@njit(cache=True)
def in_check(P):  # type: ignore[no-untyped-def]
    side = P[SIDE]
    return is_attacked(P, P[KSQ + side], 1 - side)


# ----------------------------------------------------------------------------- setup


@njit(cache=True)
def compute_hash(P):  # type: ignore[no-untyped-def]
    h = np.int64(0)
    for sq in range(64):
        pc = P[sq]
        if pc != 0:
            h ^= ZOB[pc, sq]
    if P[SIDE] == 1:
        h ^= ZOB_SIDE
    h ^= ZOB_CASTLE[P[CASTLE]]
    if P[EP] >= 0:
        h ^= ZOB_EP[P[EP] & 7]
    return h


@njit(cache=True)
def rebuild(P):  # type: ignore[no-untyped-def]
    """Recompute bitboards, occupancy, king squares, hash and material from P[0:64]."""
    for i in range(13):
        P[BB + i] = 0
    P[OCC] = 0
    P[OCC + 1] = 0
    P[NONPAWN] = 0
    for sq in range(64):
        pc = P[sq]
        if pc != 0:
            b = bit(sq)
            P[BB + pc] |= b
            P[OCC + piece_color(pc)] |= b
            pt = piece_type(pc)
            if pt == KING:
                P[KSQ + piece_color(pc)] = sq
            elif pt != PAWN:
                P[NONPAWN] += 1
    P[OCC + 2] = P[OCC] | P[OCC + 1]
    P[HASH] = compute_hash(P)
    P[LAST_PIECE] = 0
    P[LAST_CAPTURED] = 0
    P[LAST_PROMO] = 0
    P[LAST_ROOK_FROM] = -1


def new_position() -> np.ndarray:
    P = np.zeros(PSIZE, dtype=np.int64)
    P[EP] = -1
    P[LAST_ROOK_FROM] = -1
    return P


def from_board(board, P: np.ndarray | None = None) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Fill ``P`` from a python-chess Board (the legal-move authority parses the FEN)."""
    if P is None:
        P = new_position()
    P[:64] = 0
    for sq, piece in board.piece_map().items():
        P[sq] = piece.piece_type + (6 if piece.color == False else 0)  # noqa: E712
    P[SIDE] = 0 if board.turn else 1
    P[EP] = -1
    if board.ep_square is not None and board.has_legal_en_passant():
        # keep the ep square only when a capture is actually possible (matches make_move)
        P[EP] = board.ep_square
    rights = 0
    if board.has_kingside_castling_rights(True):
        rights |= CASTLE_WK
    if board.has_queenside_castling_rights(True):
        rights |= CASTLE_WQ
    if board.has_kingside_castling_rights(False):
        rights |= CASTLE_BK
    if board.has_queenside_castling_rights(False):
        rights |= CASTLE_BQ
    P[CASTLE] = rights
    P[HALF] = board.halfmove_clock
    rebuild(P)
    return P


# ----------------------------------------------------------------------------- make / unmake


@njit(cache=True)
def _move_piece(P, pc, frm, to):  # type: ignore[no-untyped-def]
    fb = bit(frm)
    tb = bit(to)
    P[BB + pc] ^= fb | tb
    P[OCC + piece_color(pc)] ^= fb | tb
    P[frm] = 0
    P[to] = pc
    P[HASH] ^= ZOB[pc, frm] ^ ZOB[pc, to]


@njit(cache=True)
def _remove_piece(P, pc, sq):  # type: ignore[no-untyped-def]
    b = bit(sq)
    P[BB + pc] ^= b
    P[OCC + piece_color(pc)] ^= b
    P[sq] = 0
    P[HASH] ^= ZOB[pc, sq]


@njit(cache=True)
def _add_piece(P, pc, sq):  # type: ignore[no-untyped-def]
    b = bit(sq)
    P[BB + pc] |= b
    P[OCC + piece_color(pc)] |= b
    P[sq] = pc
    P[HASH] ^= ZOB[pc, sq]


@njit(cache=True)
def make_move(P, undo, ply, move):  # type: ignore[no-untyped-def]
    """Apply ``move``.  Returns False (after restoring nothing) if it leaves own king in check;
    the caller must then call unmake_move.  The undo record is written to undo[ply]."""
    side = P[SIDE]
    them = 1 - side
    frm = mv_from(move)
    to = mv_to(move)
    promo = mv_promo(move)
    flags = mv_flags(move)
    pc = P[frm]
    captured = np.int64(0)
    capsq = to
    if flags & F_EP:
        capsq = to - 8 if side == WHITE else to + 8
        captured = P[capsq]
    elif flags & F_CAPTURE:
        captured = P[to]

    undo[ply, 0] = move
    undo[ply, 1] = captured
    undo[ply, 2] = P[EP]
    undo[ply, 3] = P[CASTLE]
    undo[ply, 4] = P[HALF]
    undo[ply, 5] = P[HASH]

    # hash: clear old ep / castle contributions
    if P[EP] >= 0:
        P[HASH] ^= ZOB_EP[P[EP] & 7]
    P[HASH] ^= ZOB_CASTLE[P[CASTLE]]

    if captured != 0:
        _remove_piece(P, captured, capsq)
        if piece_type(captured) != PAWN:
            P[NONPAWN] -= 1
    _move_piece(P, pc, frm, to)
    rook_from = np.int64(-1)
    rook_to = np.int64(-1)
    promo_pc = np.int64(0)
    if promo != 0:
        promo_pc = make_piece(side, promo)
        _remove_piece(P, pc, to)
        _add_piece(P, promo_pc, to)
        P[NONPAWN] += 1
    elif flags & F_CASTLE:
        if to > frm:  # kingside
            rook_from = frm + 3
            rook_to = frm + 1
        else:
            rook_from = frm - 4
            rook_to = frm - 1
        _move_piece(P, make_piece(side, ROOK), rook_from, rook_to)
    if piece_type(pc) == KING:
        P[KSQ + side] = to

    P[EP] = -1
    if flags & F_DOUBLE:
        ep = (frm + to) >> 1
        # only record ep if an enemy pawn could capture (keeps hash consistent with FEN)
        if PAWN_ATT[side, ep] & P[BB + 6 * them + PAWN]:
            P[EP] = ep
            P[HASH] ^= ZOB_EP[ep & 7]
    P[CASTLE] &= CASTLE_MASK[frm] & CASTLE_MASK[to]
    P[HASH] ^= ZOB_CASTLE[P[CASTLE]]
    P[HALF] += 1
    if captured != 0:
        P[HALF] = 0
    if piece_type(pc) == PAWN:
        P[HALF] = 0
    P[SIDE] = them
    P[HASH] ^= ZOB_SIDE
    P[OCC + 2] = P[OCC] | P[OCC + 1]

    P[LAST_PIECE] = pc
    P[LAST_FROM] = frm
    P[LAST_TO] = to
    P[LAST_CAPTURED] = captured
    P[LAST_CAPSQ] = capsq
    P[LAST_PROMO] = promo_pc
    P[LAST_ROOK_FROM] = rook_from
    P[LAST_ROOK_TO] = rook_to

    return not is_attacked(P, P[KSQ + side], them)


@njit(cache=True)
def unmake_move(P, undo, ply):  # type: ignore[no-untyped-def]
    move = undo[ply, 0]
    captured = undo[ply, 1]
    them = P[SIDE]
    side = 1 - them
    frm = mv_from(move)
    to = mv_to(move)
    promo = mv_promo(move)
    flags = mv_flags(move)
    pc = P[to]
    if promo != 0:
        _remove_piece(P, pc, to)
        pc = make_piece(side, PAWN)
        _add_piece(P, pc, to)
        P[NONPAWN] -= 1
    _move_piece(P, pc, to, frm)
    if flags & F_CASTLE:
        if to > frm:
            _move_piece(P, make_piece(side, ROOK), frm + 1, frm + 3)
        else:
            _move_piece(P, make_piece(side, ROOK), frm - 1, frm - 4)
    if captured != 0:
        capsq = to
        if flags & F_EP:
            capsq = to - 8 if side == WHITE else to + 8
        _add_piece(P, captured, capsq)
        if piece_type(captured) != PAWN:
            P[NONPAWN] += 1
    if piece_type(pc) == KING:
        P[KSQ + side] = frm
    P[EP] = undo[ply, 2]
    P[CASTLE] = undo[ply, 3]
    P[HALF] = undo[ply, 4]
    P[HASH] = undo[ply, 5]
    P[SIDE] = side
    P[OCC + 2] = P[OCC] | P[OCC + 1]


@njit(cache=True)
def make_null(P, undo, ply):  # type: ignore[no-untyped-def]
    undo[ply, 0] = 0
    undo[ply, 1] = 0
    undo[ply, 2] = P[EP]
    undo[ply, 3] = P[CASTLE]
    undo[ply, 4] = P[HALF]
    undo[ply, 5] = P[HASH]
    if P[EP] >= 0:
        P[HASH] ^= ZOB_EP[P[EP] & 7]
        P[EP] = -1
    P[SIDE] = 1 - P[SIDE]
    P[HASH] ^= ZOB_SIDE
    P[HALF] += 1


@njit(cache=True)
def unmake_null(P, undo, ply):  # type: ignore[no-untyped-def]
    P[EP] = undo[ply, 2]
    P[HALF] = undo[ply, 4]
    P[HASH] = undo[ply, 5]
    P[SIDE] = 1 - P[SIDE]


# ----------------------------------------------------------------------------- movegen


@njit(cache=True)
def _add(out, n, frm, to, promo, flags):  # type: ignore[no-untyped-def]
    out[n] = np.int64(frm) | (np.int64(to) << 6) | (np.int64(promo) << 12) | (np.int64(flags) << 16)
    return n + 1


@njit(cache=True)
def _add_promos(out, n, frm, to, flags, captures_only):  # type: ignore[no-untyped-def]
    n = _add(out, n, frm, to, QUEEN, flags)
    if not captures_only:
        n = _add(out, n, frm, to, KNIGHT, flags)
        n = _add(out, n, frm, to, ROOK, flags)
        n = _add(out, n, frm, to, BISHOP, flags)
    return n


@njit(cache=True)
def _gen_pawns(P, out, n, captures_only):  # type: ignore[no-untyped-def]
    side = P[SIDE]
    them = 1 - side
    occ = P[OCC + 2]
    enemy = P[OCC + them]
    pawns = P[BB + 6 * side + PAWN]
    if side == WHITE:
        promo_rank = RANK_MASK[7]
        pushes = (pawns << np.int64(8)) & ~occ
        dbl = ((pushes & RANK_MASK[2]) << np.int64(8)) & ~occ
        fwd = 8
    else:
        promo_rank = RANK_MASK[0]
        pushes = shr8(pawns) & ~occ
        dbl = shr8(pushes & RANK_MASK[5]) & ~occ
        fwd = -8
    pp = pushes & promo_rank
    while pp != 0:
        to = lsb(pp)
        pp &= pp - 1
        n = _add_promos(out, n, to - fwd, to, 0, captures_only)
    if not captures_only:
        pp = pushes & ~promo_rank
        while pp != 0:
            to = lsb(pp)
            pp &= pp - 1
            n = _add(out, n, to - fwd, to, 0, 0)
        while dbl != 0:
            to = lsb(dbl)
            dbl &= dbl - 1
            n = _add(out, n, to - 2 * fwd, to, 0, F_DOUBLE)
    pw = pawns
    while pw != 0:
        frm = lsb(pw)
        pw &= pw - 1
        att = PAWN_ATT[side, frm] & enemy
        while att != 0:
            to = lsb(att)
            att &= att - 1
            if (bit(to) & promo_rank) != 0:
                n = _add_promos(out, n, frm, to, F_CAPTURE, captures_only)
            else:
                n = _add(out, n, frm, to, 0, F_CAPTURE)
    ep = P[EP]
    if ep >= 0:
        att = PAWN_ATT[them, ep] & pawns
        while att != 0:
            frm = lsb(att)
            att &= att - 1
            n = _add(out, n, frm, ep, 0, F_CAPTURE | F_EP)
    return n


@njit(cache=True)
def _gen_pieces(P, out, n, captures_only):  # type: ignore[no-untyped-def]
    side = P[SIDE]
    base = 6 * side
    occ = P[OCC + 2]
    enemy = P[OCC + 1 - side]
    target = enemy if captures_only else ~P[OCC + side]
    for pt in range(KNIGHT, KING + 1):
        bbp = P[BB + base + pt]
        while bbp != 0:
            frm = lsb(bbp)
            bbp &= bbp - 1
            if pt == KNIGHT:
                att = KNIGHT_ATT[frm]
            elif pt == BISHOP:
                att = bishop_attacks(frm, occ)
            elif pt == ROOK:
                att = rook_attacks(frm, occ)
            elif pt == QUEEN:
                att = bishop_attacks(frm, occ) | rook_attacks(frm, occ)
            else:
                att = KING_ATT[frm]
            att &= target
            caps = att & enemy
            while caps != 0:
                to = lsb(caps)
                caps &= caps - 1
                n = _add(out, n, frm, to, 0, F_CAPTURE)
            quiets = att & ~enemy
            while quiets != 0:
                to = lsb(quiets)
                quiets &= quiets - 1
                n = _add(out, n, frm, to, 0, 0)
    return n


@njit(cache=True)
def _castle_ok(P, rights_bit, empty_mask, rook_sq, rook_pc, ksq, s1, s2, them):  # type: ignore[no-untyped-def]
    if (P[CASTLE] & rights_bit) == 0:
        return False
    if (P[OCC + 2] & empty_mask) != 0:
        return False
    if P[rook_sq] != rook_pc:
        return False
    if is_attacked(P, ksq, them):
        return False
    if is_attacked(P, s1, them):
        return False
    return not is_attacked(P, s2, them)


@njit(cache=True)
def _gen_castling(P, out, n):  # type: ignore[no-untyped-def]
    side = P[SIDE]
    them = 1 - side
    if side == WHITE:
        if P[KSQ] != E1:
            return n
        if _castle_ok(P, CASTLE_WK, WK_EMPTY, H1, WR, E1, F1, G1, them):
            n = _add(out, n, E1, G1, 0, F_CASTLE)
        if _castle_ok(P, CASTLE_WQ, WQ_EMPTY, A1, WR, E1, D1, C1, them):
            n = _add(out, n, E1, C1, 0, F_CASTLE)
    else:
        if P[KSQ + 1] != E8:
            return n
        if _castle_ok(P, CASTLE_BK, BK_EMPTY, H8, BR, E8, F8, G8, them):
            n = _add(out, n, E8, G8, 0, F_CASTLE)
        if _castle_ok(P, CASTLE_BQ, BQ_EMPTY, A8, BR, E8, D8, C8, them):
            n = _add(out, n, E8, C8, 0, F_CASTLE)
    return n


@njit(cache=True)
def gen_moves(P, out, captures_only):  # type: ignore[no-untyped-def]
    """Pseudo-legal moves into ``out``; returns the count.  Castling is generated fully legal
    (rights, empty path, no attacked transit squares).  With ``captures_only`` only captures and
    queen promotions are generated."""
    n = _gen_pawns(P, out, 0, captures_only)
    n = _gen_pieces(P, out, n, captures_only)
    if not captures_only:
        n = _gen_castling(P, out, n)
    return n


@njit(cache=True)
def perft(P, undo, moves, ply, depth):  # type: ignore[no-untyped-def]
    n = gen_moves(P, moves[ply], False)
    if depth == 1:
        cnt = 0
        for i in range(n):
            if make_move(P, undo, ply, moves[ply, i]):
                cnt += 1
            unmake_move(P, undo, ply)
        return cnt
    total = 0
    for i in range(n):
        if make_move(P, undo, ply, moves[ply, i]):
            total += perft(P, undo, moves, ply + 1, depth - 1)
        unmake_move(P, undo, ply)
    return total


@njit(cache=True)
def has_legal_move(P, undo, moves, ply):  # type: ignore[no-untyped-def]
    n = gen_moves(P, moves[ply], False)
    for i in range(n):
        ok = make_move(P, undo, ply, moves[ply, i])
        unmake_move(P, undo, ply)
        if ok:
            return True
    return False


def move_to_uci(move: int) -> str:
    frm = int(move & 63)
    to = int((move >> 6) & 63)
    promo = int((move >> 12) & 7)
    s = "abcdefgh"[frm % 8] + str(frm // 8 + 1) + "abcdefgh"[to % 8] + str(to // 8 + 1)
    if promo:
        s += "nbrq"[promo - 2]
    return s


def new_undo() -> np.ndarray:
    return np.zeros((MAX_PLY + 2, UNDO_W), dtype=np.int64)


def new_movelists() -> np.ndarray:
    return np.zeros((MAX_PLY + 2, MAX_MOVES), dtype=np.int64)
