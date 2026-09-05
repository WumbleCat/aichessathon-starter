"""numba bitboard position: tables, move generation, make/unmake, perft.

Everything here is written from scratch for this project. The only external input is
python-chess, used at import time to build attack tables (knight, king, pawn and the
edge-stripped sliding-attack subsets) and, in tests, as the legality oracle.

Representation (all int64 numpy arrays so numba never mixes signed/unsigned):

    tab[]       one flat int64 lookup table (attack tables, Zobrist keys, PST); passed as
                an argument because numba embeds *global* arrays as compile-time constants,
                which costs ~8 s of compile per function for a 64x512 table
    bb[13]      bitboard per piece code (0 unused; 1..6 white P N B R Q K; 7..12 black)
    occ[3]      white, black, all
    sq[64]      mailbox of piece codes
    st[16]      scalar state, see ST_* below
    undo[N][10] per-ply undo records, see U_* below

Move encoding (int64):

    bits 0-5 from, 6-11 to, 12-14 promotion piece type (0 none, 2..5 N B R Q),
    16 capture, 17 double push, 18 en passant, 19 castle,
    20-23 moving piece code, 24-27 captured piece code (pawn for en passant).
"""

from __future__ import annotations

import chess
import numpy as np
from numba import boolean, int64, njit, void

A1 = int64[::1]
A2 = int64[:, ::1]

# ----------------------------------------------------------------------------- constants

EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 0, 1

ST_SIDE, ST_CASTLE, ST_EP, ST_HALF, ST_HASH, ST_PLY, ST_MG, ST_EG, ST_PHASE, ST_FULLMOVE = range(10)
ST_SIZE = 16

U_MOVE, U_CAPT, U_CASTLE, U_EP, U_HALF, U_HASH, U_MG, U_EG, U_PHASE = range(9)
U_SIZE = 10

MAX_PLY = 128
MOVE_BUF = 256

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

FLAG_CAPTURE = 1 << 16
FLAG_DOUBLE = 1 << 17
FLAG_EP = 1 << 18
FLAG_CASTLE = 1 << 19

# ------------------------------------------------------------------------------- tables


def _bb(values: list[int]) -> np.ndarray:
    """Python ints (possibly >= 2**63) -> int64 array with the same bit patterns."""
    return np.array(values, dtype=np.uint64).view(np.int64)


def _subset_table(masks: list[int], attacks: list[dict[int, int]], width: int) -> tuple:
    mask_arr = _bb(masks)
    table = np.zeros((64, width), dtype=np.uint64)
    for s in range(64):
        bits = [b for b in range(64) if masks[s] >> b & 1]
        for idx in range(1 << len(bits)):
            occ = 0
            for i, b in enumerate(bits):
                if idx >> i & 1:
                    occ |= 1 << b
            table[s, idx] = attacks[s][occ]
    return mask_arr, table.view(np.int64)


_RANK_MASK, _RANK_ATT = _subset_table(list(chess.BB_RANK_MASKS), list(chess.BB_RANK_ATTACKS), 64)
_FILE_MASK, _FILE_ATT = _subset_table(list(chess.BB_FILE_MASKS), list(chess.BB_FILE_ATTACKS), 64)
_DIAG_MASK, _DIAG_ATT = _subset_table(list(chess.BB_DIAG_MASKS), list(chess.BB_DIAG_ATTACKS), 512)

DEBRUIJN = 0x03F79D71B4CB0A89
_index64 = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _index64[((1 << _i) * DEBRUIJN & (2**64 - 1)) >> 58] = _i
INDEX64 = _index64

# castling right bits cleared when a piece moves from / to a square
_cm = np.full(64, 15, dtype=np.int64)
_cm[chess.E1] = 15 & ~(CASTLE_WK | CASTLE_WQ)
_cm[chess.H1] = 15 & ~CASTLE_WK
_cm[chess.A1] = 15 & ~CASTLE_WQ
_cm[chess.E8] = 15 & ~(CASTLE_BK | CASTLE_BQ)
_cm[chess.H8] = 15 & ~CASTLE_BK
_cm[chess.A8] = 15 & ~CASTLE_BQ

# Zobrist keys, fixed seed so hashes are reproducible between runs
_rng = np.random.default_rng(20240904)
_ZOB_PIECE = _rng.integers(0, 2**63 - 1, size=(13, 64), dtype=np.int64)
_ZOB_CASTLE = _rng.integers(0, 2**63 - 1, size=16, dtype=np.int64)
_ZOB_EP = _rng.integers(0, 2**63 - 1, size=8, dtype=np.int64)
ZOB_SIDE = int(_rng.integers(0, 2**63 - 1))

# ------------------------------------------------------------------- evaluation tables
# Material and piece-square tables (white point of view, a1 = index 0). Written by hand
# for this project; midgame and endgame variants are blended by game phase.

MAT_MG = np.array([0, 100, 325, 335, 500, 975, 0], dtype=np.int64)
MAT_EG = np.array([0, 125, 305, 325, 550, 1000, 0], dtype=np.int64)
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0], dtype=np.int64)
PHASE_MAX = 24

# fmt: off
_PST_MG = {
    PAWN: [
        0,   0,   0,   0,   0,   0,   0,   0,
       -8,  -2,  -6, -20, -20,  10,  10,  -6,
       -6,  -4,  -2,   2,   4,   0,   4,  -8,
       -4,   0,   8,  18,  20,   6,  -2,  -6,
        2,   6,  14,  24,  26,  14,   4,   0,
       12,  20,  30,  36,  36,  30,  20,  12,
       50,  55,  60,  60,  60,  60,  55,  50,
        0,   0,   0,   0,   0,   0,   0,   0,
    ],
    KNIGHT: [
      -60, -40, -30, -25, -25, -30, -40, -60,
      -35, -20,   0,   4,   4,   0, -20, -35,
      -25,   4,  12,  16,  16,  12,   4, -25,
      -20,   8,  18,  24,  24,  18,   8, -20,
      -20,  10,  20,  26,  26,  20,  10, -20,
      -25,   6,  16,  22,  22,  16,   6, -25,
      -35, -15,   2,   8,   8,   2, -15, -35,
      -70, -40, -25, -20, -20, -25, -40, -70,
    ],
    BISHOP: [
      -25, -10, -14, -10, -10, -14, -10, -25,
       -8,  12,   2,   6,   6,   2,  12,  -8,
       -4,  10,  12,  10,  10,  12,  10,  -4,
       -4,   6,  14,  18,  18,  14,   6,  -4,
       -4,   4,  12,  18,  18,  12,   4,  -4,
       -6,   6,  10,  12,  12,  10,   6,  -6,
      -10,   2,   2,   2,   2,   2,   2, -10,
      -20, -10, -12, -10, -10, -12, -10, -20,
    ],
    ROOK: [
       -4,  -2,   4,  10,  10,   4,  -2,  -4,
      -12,  -6,  -2,   4,   4,  -2,  -6, -12,
      -10,  -4,  -2,   0,   0,  -2,  -4, -10,
       -8,  -4,   0,   2,   2,   0,  -4,  -8,
       -6,  -2,   2,   4,   4,   2,  -2,  -6,
       -2,   2,   6,   8,   8,   6,   2,  -2,
       10,  14,  18,  20,  20,  18,  14,  10,
        6,   8,  10,  12,  12,  10,   8,   6,
    ],
    QUEEN: [
      -20, -12, -10,   0, -10, -12, -14, -20,
      -12,  -4,   2,   4,   4,   2,  -4, -12,
       -8,   2,   6,   6,   6,   6,   2,  -8,
       -4,   2,   6,   8,   8,   6,   2,  -4,
       -4,   2,   6,   8,   8,   6,   2,  -4,
       -8,   2,   6,   6,   6,   6,   2,  -8,
      -12,  -4,   2,   4,   4,   2,  -4, -12,
      -20, -12, -10,  -4,  -4, -10, -12, -20,
    ],
    KING: [
       20,  35,  10, -10,  -5,   0,  35,  20,
       15,  15,  -5, -20, -20,  -5,  15,  15,
      -20, -30, -35, -45, -45, -35, -30, -20,
      -35, -45, -50, -60, -60, -50, -45, -35,
      -45, -55, -60, -70, -70, -60, -55, -45,
      -50, -60, -65, -75, -75, -65, -60, -50,
      -55, -65, -70, -80, -80, -70, -65, -55,
      -60, -70, -75, -85, -85, -75, -70, -60,
    ],
}
_PST_EG = {
    PAWN: [
        0,   0,   0,   0,   0,   0,   0,   0,
        4,   4,   2,   2,   2,   2,   4,   4,
        4,   4,   2,   0,   0,   2,   4,   4,
        8,   8,   4,   0,   0,   4,   8,   8,
       18,  16,  12,   8,   8,  12,  16,  18,
       40,  38,  32,  26,  26,  32,  38,  40,
       80,  78,  72,  66,  66,  72,  78,  80,
        0,   0,   0,   0,   0,   0,   0,   0,
    ],
    KNIGHT: [
      -45, -35, -25, -20, -20, -25, -35, -45,
      -30, -15,  -5,   0,   0,  -5, -15, -30,
      -20,  -5,   8,  12,  12,   8,  -5, -20,
      -15,   0,  12,  18,  18,  12,   0, -15,
      -15,   0,  12,  18,  18,  12,   0, -15,
      -20,  -5,   8,  12,  12,   8,  -5, -20,
      -30, -15,  -5,   0,   0,  -5, -15, -30,
      -45, -35, -25, -20, -20, -25, -35, -45,
    ],
    BISHOP: [
      -20, -10,  -8,  -6,  -6,  -8, -10, -20,
      -10,  -2,   2,   4,   4,   2,  -2, -10,
       -8,   2,   8,  10,  10,   8,   2,  -8,
       -6,   4,  10,  14,  14,  10,   4,  -6,
       -6,   4,  10,  14,  14,  10,   4,  -6,
       -8,   2,   8,  10,  10,   8,   2,  -8,
      -10,  -2,   2,   4,   4,   2,  -2, -10,
      -20, -10,  -8,  -6,  -6,  -8, -10, -20,
    ],
    ROOK: [
       -2,   0,   2,   2,   2,   2,   0,  -2,
       -4,  -2,   0,   0,   0,   0,  -2,  -4,
       -4,  -2,   0,   0,   0,   0,  -2,  -4,
       -2,   0,   2,   2,   2,   2,   0,  -2,
        0,   2,   4,   4,   4,   4,   2,   0,
        2,   4,   6,   6,   6,   6,   4,   2,
        6,   8,  10,  10,  10,  10,   8,   6,
        4,   6,   8,   8,   8,   8,   6,   4,
    ],
    QUEEN: [
      -25, -18, -14, -10, -10, -14, -18, -25,
      -16,  -8,  -4,   0,   0,  -4,  -8, -16,
      -12,  -2,   6,  10,  10,   6,  -2, -12,
       -8,   2,  10,  16,  16,  10,   2,  -8,
       -8,   2,  10,  16,  16,  10,   2,  -8,
      -12,  -2,   6,  10,  10,   6,  -2, -12,
      -16,  -8,  -4,   0,   0,  -4,  -8, -16,
      -25, -18, -14, -10, -10, -14, -18, -25,
    ],
    KING: [
      -60, -40, -30, -25, -25, -30, -40, -60,
      -35, -15,  -5,   0,   0,  -5, -15, -35,
      -25,  -5,  12,  20,  20,  12,  -5, -25,
      -20,   0,  20,  30,  30,  20,   0, -20,
      -20,   0,  22,  32,  32,  22,   0, -20,
      -25,  -5,  15,  25,  25,  15,  -5, -25,
      -35, -15,  -2,   5,   5,  -2, -15, -35,
      -60, -40, -30, -25, -25, -30, -40, -60,
    ],
}
# fmt: on


def _build_pst() -> tuple[np.ndarray, np.ndarray]:
    mg = np.zeros((13, 64), dtype=np.int64)
    eg = np.zeros((13, 64), dtype=np.int64)
    for pt in range(PAWN, KING + 1):
        for s in range(64):
            wmg = MAT_MG[pt] + _PST_MG[pt][s]
            weg = MAT_EG[pt] + _PST_EG[pt][s]
            mg[pt, s] = wmg
            eg[pt, s] = weg
            mirror = s ^ 56
            mg[pt + 6, mirror] = -wmg
            eg[pt + 6, mirror] = -weg
    return mg, eg


PST_MG, PST_EG = _build_pst()

# ------------------------------------------------------------------ flat table layout
# Every jitted function receives ``tab`` and indexes it with these offsets.

_parts: list[tuple[str, np.ndarray]] = [
    ("T_KNIGHT", _bb([chess.BB_KNIGHT_ATTACKS[s] for s in range(64)])),
    ("T_KING", _bb([chess.BB_KING_ATTACKS[s] for s in range(64)])),
    ("T_PAWN", np.concatenate([  # side * 64 + square
        _bb([chess.BB_PAWN_ATTACKS[chess.WHITE][s] for s in range(64)]),
        _bb([chess.BB_PAWN_ATTACKS[chess.BLACK][s] for s in range(64)]),
    ])),
    ("T_FILE", _bb([chess.BB_FILES[f] for f in range(8)])),
    ("T_RANK", _bb([chess.BB_RANKS[r] for r in range(8)])),
    ("T_RANK_MASK", _RANK_MASK),
    ("T_FILE_MASK", _FILE_MASK),
    ("T_DIAG_MASK", _DIAG_MASK),
    ("T_RANK_ATT", _RANK_ATT.ravel()),  # square * 64 + index
    ("T_FILE_ATT", _FILE_ATT.ravel()),  # square * 64 + index
    ("T_DIAG_ATT", _DIAG_ATT.ravel()),  # square * 512 + index
    ("T_CASTLE_MASK", _cm),
    ("T_ZOB_PIECE", _ZOB_PIECE.ravel()),  # piece * 64 + square
    ("T_ZOB_CASTLE", _ZOB_CASTLE),
    ("T_ZOB_EP", _ZOB_EP),
    ("T_PST_MG", PST_MG.ravel()),  # piece * 64 + square
    ("T_PST_EG", PST_EG.ravel()),
    ("T_PHASE", PHASE_INC),
]
_offsets: dict[str, int] = {}
_pos = 0
for _name, _arr in _parts:
    _offsets[_name] = _pos
    _pos += int(_arr.size)
TAB = np.concatenate([a.astype(np.int64) for _, a in _parts])
T_KNIGHT = _offsets["T_KNIGHT"]
T_KING = _offsets["T_KING"]
T_PAWN = _offsets["T_PAWN"]
T_FILE = _offsets["T_FILE"]
T_RANK = _offsets["T_RANK"]
T_RANK_MASK = _offsets["T_RANK_MASK"]
T_FILE_MASK = _offsets["T_FILE_MASK"]
T_DIAG_MASK = _offsets["T_DIAG_MASK"]
T_RANK_ATT = _offsets["T_RANK_ATT"]
T_FILE_ATT = _offsets["T_FILE_ATT"]
T_DIAG_ATT = _offsets["T_DIAG_ATT"]
T_CASTLE_MASK = _offsets["T_CASTLE_MASK"]
T_ZOB_PIECE = _offsets["T_ZOB_PIECE"]
T_ZOB_CASTLE = _offsets["T_ZOB_CASTLE"]
T_ZOB_EP = _offsets["T_ZOB_EP"]
T_PST_MG = _offsets["T_PST_MG"]
T_PST_EG = _offsets["T_PST_EG"]
T_PHASE = _offsets["T_PHASE"]

# ----------------------------------------------------------------------------- bit utils


@njit(int64(int64), cache=False)
def bit(s: int) -> int:
    return 1 << s


@njit(int64(int64), cache=False)
def lsb(b: int) -> int:
    return INDEX64[(((b & -b) * DEBRUIJN) >> 58) & 63]


@njit(int64(int64), cache=False)
def popcount(b: int) -> int:
    b = b - ((b >> 1) & 0x5555555555555555)
    b = (b & 0x3333333333333333) + ((b >> 2) & 0x3333333333333333)
    b = (b + (b >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((b * 0x0101010101010101) >> 56) & 0x7F


@njit(int64(int64, int64), cache=False)
def pext(occ: int, mask: int) -> int:
    idx = 0
    b = 1
    while mask != 0:
        low = mask & -mask
        if occ & low:
            idx |= b
        b <<= 1
        mask ^= low
    return idx


@njit(int64(A1, int64, int64), cache=False)
def rook_attacks(tab, s: int, occ: int) -> int:
    return tab[T_RANK_ATT + (s << 6) + pext(occ, tab[T_RANK_MASK + s])] | tab[
        T_FILE_ATT + (s << 6) + pext(occ, tab[T_FILE_MASK + s])
    ]


@njit(int64(A1, int64, int64), cache=False)
def bishop_attacks(tab, s: int, occ: int) -> int:
    return tab[T_DIAG_ATT + (s << 9) + pext(occ, tab[T_DIAG_MASK + s])]


@njit(boolean(A1, A1, int64, int64, int64), cache=False)
def attacked(tab, bb, occ_all: int, s: int, by: int) -> bool:
    """True when side ``by`` (0 white, 1 black) attacks square ``s``."""
    base = 6 * by
    if tab[T_PAWN + ((1 - by) << 6) + s] & bb[base + PAWN]:
        return True
    if tab[T_KNIGHT + s] & bb[base + KNIGHT]:
        return True
    if tab[T_KING + s] & bb[base + KING]:
        return True
    bq = bb[base + BISHOP] | bb[base + QUEEN]
    if bq and bishop_attacks(tab, s, occ_all) & bq:
        return True
    rq = bb[base + ROOK] | bb[base + QUEEN]
    return bool(rq and rook_attacks(tab, s, occ_all) & rq)


@njit(int64(A1, A1, int64, int64), cache=False)
def attackers_to(tab, bb, occ_all: int, s: int) -> int:
    """Bitboard of all pieces of both colours attacking ``s`` under occupancy occ_all."""
    a = tab[T_PAWN + (BLACK << 6) + s] & bb[PAWN]
    a |= tab[T_PAWN + (WHITE << 6) + s] & bb[6 + PAWN]
    a |= tab[T_KNIGHT + s] & (bb[KNIGHT] | bb[6 + KNIGHT])
    a |= tab[T_KING + s] & (bb[KING] | bb[6 + KING])
    bq = bb[BISHOP] | bb[QUEEN] | bb[6 + BISHOP] | bb[6 + QUEEN]
    a |= bishop_attacks(tab, s, occ_all) & bq
    rq = bb[ROOK] | bb[QUEEN] | bb[6 + ROOK] | bb[6 + QUEEN]
    a |= rook_attacks(tab, s, occ_all) & rq
    return a & occ_all


@njit(int64(A1, int64), cache=False)
def king_square(bb, side: int) -> int:
    return lsb(bb[6 * side + KING])


@njit(boolean(A1, A1, A1, A1), cache=False)
def in_check(tab, bb, occ, st) -> bool:
    side = st[ST_SIDE]
    return attacked(tab, bb, occ[2], king_square(bb, side), 1 - side)


# ----------------------------------------------------------------------- move encoding


@njit(int64(int64), cache=False)
def move_from(m: int) -> int:
    return m & 63


@njit(int64(int64), cache=False)
def move_to(m: int) -> int:
    return (m >> 6) & 63


@njit(int64(int64), cache=False)
def move_promo(m: int) -> int:
    return (m >> 12) & 7


@njit(int64(int64), cache=False)
def move_piece(m: int) -> int:
    return (m >> 20) & 15


@njit(int64(int64), cache=False)
def move_captured(m: int) -> int:
    return (m >> 24) & 15


@njit(int64(int64, int64, int64, int64, int64, int64), cache=False)
def encode(frm: int, to: int, promo: int, flags: int, piece: int, captured: int) -> int:
    return frm | (to << 6) | (promo << 12) | flags | (piece << 20) | (captured << 24)


# -------------------------------------------------------------------- piece placement


@njit(void(A1, A1, A1, A1, A1, int64, int64), cache=False)
def put_piece(tab, bb, occ, sq, st, piece: int, s: int) -> None:
    b = bit(s)
    bb[piece] |= b
    occ[(piece - 1) // 6] |= b
    occ[2] |= b
    sq[s] = piece
    idx = (piece << 6) + s
    st[ST_HASH] ^= tab[T_ZOB_PIECE + idx]
    st[ST_MG] += tab[T_PST_MG + idx]
    st[ST_EG] += tab[T_PST_EG + idx]
    st[ST_PHASE] += tab[T_PHASE + piece]


@njit(void(A1, A1, A1, A1, A1, int64, int64), cache=False)
def remove_piece(tab, bb, occ, sq, st, piece: int, s: int) -> None:
    b = bit(s)
    bb[piece] ^= b
    occ[(piece - 1) // 6] ^= b
    occ[2] ^= b
    sq[s] = EMPTY
    idx = (piece << 6) + s
    st[ST_HASH] ^= tab[T_ZOB_PIECE + idx]
    st[ST_MG] -= tab[T_PST_MG + idx]
    st[ST_EG] -= tab[T_PST_EG + idx]
    st[ST_PHASE] -= tab[T_PHASE + piece]


@njit(void(A1, A1, A1, int64, int64), cache=False)
def _lift(bb, occ, sq, piece: int, s: int) -> None:
    b = bit(s)
    bb[piece] ^= b
    occ[(piece - 1) // 6] ^= b
    occ[2] ^= b
    sq[s] = EMPTY


@njit(void(A1, A1, A1, int64, int64), cache=False)
def _drop(bb, occ, sq, piece: int, s: int) -> None:
    b = bit(s)
    bb[piece] |= b
    occ[(piece - 1) // 6] |= b
    occ[2] |= b
    sq[s] = piece


# ------------------------------------------------------------------------ make / unmake


@njit(void(A1, A1, A1, A1, A1, A2, int64), cache=False)
def make_move(tab, bb, occ, sq, st, undo, m: int) -> None:
    ply = st[ST_PLY]
    u = undo[ply]
    u[U_MOVE] = m
    u[U_CASTLE] = st[ST_CASTLE]
    u[U_EP] = st[ST_EP]
    u[U_HALF] = st[ST_HALF]
    u[U_HASH] = st[ST_HASH]
    u[U_MG] = st[ST_MG]
    u[U_EG] = st[ST_EG]
    u[U_PHASE] = st[ST_PHASE]

    side = st[ST_SIDE]
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    piece = (m >> 20) & 15
    captured = (m >> 24) & 15
    u[U_CAPT] = captured

    if st[ST_EP] >= 0:
        st[ST_HASH] ^= tab[T_ZOB_EP + (st[ST_EP] & 7)]
        st[ST_EP] = -1

    if captured != EMPTY:
        if m & FLAG_EP:
            cap_sq = to - 8 if side == WHITE else to + 8
            remove_piece(tab, bb, occ, sq, st, captured, cap_sq)
        else:
            remove_piece(tab, bb, occ, sq, st, captured, to)

    remove_piece(tab, bb, occ, sq, st, piece, frm)
    if promo:
        put_piece(tab, bb, occ, sq, st, promo + 6 * side, to)
    else:
        put_piece(tab, bb, occ, sq, st, piece, to)

    if m & FLAG_CASTLE:
        rook = ROOK + 6 * side
        if to > frm:  # king side
            remove_piece(tab, bb, occ, sq, st, rook, to + 1)
            put_piece(tab, bb, occ, sq, st, rook, to - 1)
        else:
            remove_piece(tab, bb, occ, sq, st, rook, to - 2)
            put_piece(tab, bb, occ, sq, st, rook, to + 1)

    if m & FLAG_DOUBLE:
        ep = (frm + to) >> 1
        # only record the en-passant square when an enemy pawn could actually take
        if tab[T_PAWN + (side << 6) + ep] & bb[PAWN + 6 * (1 - side)]:
            st[ST_EP] = ep
            st[ST_HASH] ^= tab[T_ZOB_EP + (ep & 7)]

    old_castle = st[ST_CASTLE]
    new_castle = old_castle & tab[T_CASTLE_MASK + frm] & tab[T_CASTLE_MASK + to]
    if new_castle != old_castle:
        st[ST_CASTLE] = new_castle
        st[ST_HASH] ^= tab[T_ZOB_CASTLE + old_castle] ^ tab[T_ZOB_CASTLE + new_castle]

    if captured != EMPTY or piece == PAWN + 6 * side:
        st[ST_HALF] = 0
    else:
        st[ST_HALF] += 1

    st[ST_SIDE] = 1 - side
    st[ST_HASH] ^= ZOB_SIDE
    st[ST_PLY] = ply + 1


@njit(void(A1, A1, A1, A1, A2), cache=False)
def unmake_move(bb, occ, sq, st, undo) -> None:
    ply = st[ST_PLY] - 1
    u = undo[ply]
    m = u[U_MOVE]
    side = 1 - st[ST_SIDE]  # the side that made the move
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    piece = (m >> 20) & 15
    captured = u[U_CAPT]

    if promo:
        _lift(bb, occ, sq, promo + 6 * side, to)
    else:
        _lift(bb, occ, sq, piece, to)
    _drop(bb, occ, sq, piece, frm)
    if captured != EMPTY:
        if m & FLAG_EP:
            cap_sq = to - 8 if side == WHITE else to + 8
            _drop(bb, occ, sq, captured, cap_sq)
        else:
            _drop(bb, occ, sq, captured, to)
    if m & FLAG_CASTLE:
        rook = ROOK + 6 * side
        if to > frm:
            _lift(bb, occ, sq, rook, to - 1)
            _drop(bb, occ, sq, rook, to + 1)
        else:
            _lift(bb, occ, sq, rook, to + 1)
            _drop(bb, occ, sq, rook, to - 2)

    st[ST_SIDE] = side
    st[ST_CASTLE] = u[U_CASTLE]
    st[ST_EP] = u[U_EP]
    st[ST_HALF] = u[U_HALF]
    st[ST_HASH] = u[U_HASH]
    st[ST_MG] = u[U_MG]
    st[ST_EG] = u[U_EG]
    st[ST_PHASE] = u[U_PHASE]
    st[ST_PLY] = ply


@njit(void(A1, A1, A2), cache=False)
def make_null(tab, st, undo) -> None:
    ply = st[ST_PLY]
    u = undo[ply]
    u[U_MOVE] = 0
    u[U_EP] = st[ST_EP]
    u[U_HASH] = st[ST_HASH]
    u[U_HALF] = st[ST_HALF]
    if st[ST_EP] >= 0:
        st[ST_HASH] ^= tab[T_ZOB_EP + (st[ST_EP] & 7)]
        st[ST_EP] = -1
    st[ST_SIDE] = 1 - st[ST_SIDE]
    st[ST_HASH] ^= ZOB_SIDE
    st[ST_HALF] += 1
    st[ST_PLY] = ply + 1


@njit(void(A1, A2), cache=False)
def unmake_null(st, undo) -> None:
    ply = st[ST_PLY] - 1
    u = undo[ply]
    st[ST_SIDE] = 1 - st[ST_SIDE]
    st[ST_EP] = u[U_EP]
    st[ST_HASH] = u[U_HASH]
    st[ST_HALF] = u[U_HALF]
    st[ST_PLY] = ply


# ------------------------------------------------------------------- move generation


@njit(int64(A1, A1, A1, A1, A1, A1, int64, boolean), cache=False)
def gen_pawn_moves(tab, bb, occ, sq, st, out, n: int, captures_only: bool) -> int:
    side = st[ST_SIDE]
    them = occ[1 - side]
    empty = ~occ[2]
    pawn = PAWN + 6 * side
    ep = st[ST_EP]
    pawn_att_base = T_PAWN + (side << 6)
    if side == WHITE:
        promo_rank = tab[T_RANK + 7]
        start_rank = tab[T_RANK + 1]
        fwd = 8
    else:
        promo_rank = tab[T_RANK + 0]
        start_rank = tab[T_RANK + 6]
        fwd = -8
    pawns = bb[pawn]
    while pawns:
        frm = lsb(pawns)
        pawns &= pawns - 1
        fb = bit(frm)
        to = frm + fwd
        tb = bit(to)
        att = tab[pawn_att_base + frm] & them
        while att:
            t = lsb(att)
            att &= att - 1
            cap = sq[t]
            if bit(t) & promo_rank:
                out[n] = encode(frm, t, QUEEN, FLAG_CAPTURE, pawn, cap)
                n += 1
                if not captures_only:
                    out[n] = encode(frm, t, KNIGHT, FLAG_CAPTURE, pawn, cap)
                    out[n + 1] = encode(frm, t, ROOK, FLAG_CAPTURE, pawn, cap)
                    out[n + 2] = encode(frm, t, BISHOP, FLAG_CAPTURE, pawn, cap)
                    n += 3
            else:
                out[n] = encode(frm, t, 0, FLAG_CAPTURE, pawn, cap)
                n += 1
        if ep >= 0 and (tab[pawn_att_base + frm] & bit(ep)):
            out[n] = encode(frm, ep, 0, FLAG_CAPTURE | FLAG_EP, pawn, PAWN + 6 * (1 - side))
            n += 1
        if tb & empty:
            if tb & promo_rank:
                out[n] = encode(frm, to, QUEEN, 0, pawn, 0)
                n += 1
                if not captures_only:
                    out[n] = encode(frm, to, KNIGHT, 0, pawn, 0)
                    out[n + 1] = encode(frm, to, ROOK, 0, pawn, 0)
                    out[n + 2] = encode(frm, to, BISHOP, 0, pawn, 0)
                    n += 3
            elif not captures_only:
                out[n] = encode(frm, to, 0, 0, pawn, 0)
                n += 1
                if fb & start_rank:
                    to2 = to + fwd
                    if bit(to2) & empty:
                        out[n] = encode(frm, to2, 0, FLAG_DOUBLE, pawn, 0)
                        n += 1
    return n


@njit(int64(A1, A1, A1, A1, A1, A1, int64, boolean), cache=False)
def gen_piece_moves(tab, bb, occ, sq, st, out, n: int, captures_only: bool) -> int:
    side = st[ST_SIDE]
    base = 6 * side
    all_occ = occ[2]
    target = occ[1 - side] if captures_only else ~occ[side]
    for piece in range(KNIGHT + base, KING + base + 1):
        pieces = bb[piece]
        pt = piece - base
        while pieces:
            frm = lsb(pieces)
            pieces &= pieces - 1
            if pt == KNIGHT:
                att = tab[T_KNIGHT + frm]
            elif pt == BISHOP:
                att = bishop_attacks(tab, frm, all_occ)
            elif pt == ROOK:
                att = rook_attacks(tab, frm, all_occ)
            elif pt == QUEEN:
                att = rook_attacks(tab, frm, all_occ) | bishop_attacks(tab, frm, all_occ)
            else:
                att = tab[T_KING + frm]
            att &= target
            while att:
                t = lsb(att)
                att &= att - 1
                cap = sq[t]
                out[n] = encode(frm, t, 0, FLAG_CAPTURE if cap else 0, piece, cap)
                n += 1
    return n


@njit(int64(A1, A1, A1, A1, A1, A1, int64), cache=False)
def gen_castling(tab, bb, occ, sq, st, out, n: int) -> int:
    side = st[ST_SIDE]
    all_occ = occ[2]
    rights = st[ST_CASTLE]
    piece = KING + 6 * side
    if side == WHITE:
        if (
            rights & CASTLE_WK
            and sq[4] == KING
            and sq[7] == ROOK
            and not (all_occ & 0x60)
            and not attacked(tab, bb, all_occ, 4, BLACK)
            and not attacked(tab, bb, all_occ, 5, BLACK)
        ):
            out[n] = encode(4, 6, 0, FLAG_CASTLE, piece, 0)
            n += 1
        if (
            rights & CASTLE_WQ
            and sq[4] == KING
            and sq[0] == ROOK
            and not (all_occ & 0x0E)
            and not attacked(tab, bb, all_occ, 4, BLACK)
            and not attacked(tab, bb, all_occ, 3, BLACK)
        ):
            out[n] = encode(4, 2, 0, FLAG_CASTLE, piece, 0)
            n += 1
    else:
        if (
            rights & CASTLE_BK
            and sq[60] == KING + 6
            and sq[63] == ROOK + 6
            and not (all_occ & (bit(61) | bit(62)))
            and not attacked(tab, bb, all_occ, 60, WHITE)
            and not attacked(tab, bb, all_occ, 61, WHITE)
        ):
            out[n] = encode(60, 62, 0, FLAG_CASTLE, piece, 0)
            n += 1
        if (
            rights & CASTLE_BQ
            and sq[60] == KING + 6
            and sq[56] == ROOK + 6
            and not (all_occ & (bit(57) | bit(58) | bit(59)))
            and not attacked(tab, bb, all_occ, 60, WHITE)
            and not attacked(tab, bb, all_occ, 59, WHITE)
        ):
            out[n] = encode(60, 58, 0, FLAG_CASTLE, piece, 0)
            n += 1
    return n


@njit(int64(A1, A1, A1, A1, A1, A1, int64, boolean), cache=False)
def gen_moves(tab, bb, occ, sq, st, out, offset: int, captures_only: bool) -> int:
    """Write pseudo-legal moves into out[offset:], return the count.

    captures_only: captures, en passant and queen promotions only (quiescence).
    Castling is generated only when the king is not in check and does not pass
    through an attacked square; the destination square is validated like any move.
    """
    n = gen_pawn_moves(tab, bb, occ, sq, st, out, offset, captures_only)
    n = gen_piece_moves(tab, bb, occ, sq, st, out, n, captures_only)
    if not captures_only:
        n = gen_castling(tab, bb, occ, sq, st, out, n)
    return n - offset


@njit(boolean(A1, A1, A1, A1), cache=False)
def is_legal_after_make(tab, bb, occ, st) -> bool:
    """After make_move: True when the mover's king is not left in check."""
    mover = 1 - st[ST_SIDE]
    return not attacked(tab, bb, occ[2], king_square(bb, mover), st[ST_SIDE])


@njit(int64(A1, A1, A1, A1, A1, A2, A1, int64), cache=False)
def perft(tab, bb, occ, sq, st, undo, moves, depth: int) -> int:
    ply = st[ST_PLY]
    n = gen_moves(tab, bb, occ, sq, st, moves, ply * MOVE_BUF, False)
    if depth == 1:
        count = 0
        for i in range(n):
            make_move(tab, bb, occ, sq, st, undo, moves[ply * MOVE_BUF + i])
            if is_legal_after_make(tab, bb, occ, st):
                count += 1
            unmake_move(bb, occ, sq, st, undo)
        return count
    total = 0
    for i in range(n):
        make_move(tab, bb, occ, sq, st, undo, moves[ply * MOVE_BUF + i])
        if is_legal_after_make(tab, bb, occ, st):
            total += perft(tab, bb, occ, sq, st, undo, moves, depth - 1)
        unmake_move(bb, occ, sq, st, undo)
    return total


# --------------------------------------------------------------------------- Python side


class Position:
    """Owns the numpy arrays for one position and offers Python-level helpers."""

    def __init__(self, fen: str = chess.STARTING_FEN) -> None:
        self.tab = TAB
        self.bb = np.zeros(13, dtype=np.int64)
        self.occ = np.zeros(3, dtype=np.int64)
        self.sq = np.zeros(64, dtype=np.int64)
        self.st = np.zeros(ST_SIZE, dtype=np.int64)
        self.undo = np.zeros((MAX_PLY + 8, U_SIZE), dtype=np.int64)
        self.moves = np.zeros((MAX_PLY + 8) * MOVE_BUF, dtype=np.int64)
        self.set_fen(fen)

    def set_fen(self, fen: str) -> None:
        board = chess.Board(fen)
        self.bb[:] = 0
        self.occ[:] = 0
        self.sq[:] = 0
        self.st[:] = 0
        self.st[ST_EP] = -1
        for s, piece in board.piece_map().items():
            code = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
            put_piece(self.tab, self.bb, self.occ, self.sq, self.st, code, s)
        self.st[ST_SIDE] = WHITE if board.turn == chess.WHITE else BLACK
        rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            rights |= CASTLE_WK
        if board.has_queenside_castling_rights(chess.WHITE):
            rights |= CASTLE_WQ
        if board.has_kingside_castling_rights(chess.BLACK):
            rights |= CASTLE_BK
        if board.has_queenside_castling_rights(chess.BLACK):
            rights |= CASTLE_BQ
        self.st[ST_CASTLE] = rights
        self.st[ST_HASH] ^= TAB[T_ZOB_CASTLE + rights]
        if board.ep_square is not None and board.has_legal_en_passant():
            self.st[ST_EP] = board.ep_square
            self.st[ST_HASH] ^= TAB[T_ZOB_EP + board.ep_square % 8]
        if board.turn == chess.BLACK:
            self.st[ST_HASH] ^= ZOB_SIDE
        self.st[ST_HALF] = board.halfmove_clock
        self.st[ST_FULLMOVE] = board.fullmove_number
        self.st[ST_PLY] = 0

    def legal_moves(self) -> list[int]:
        n = gen_moves(self.tab, self.bb, self.occ, self.sq, self.st, self.moves, 0, False)
        legal = []
        for i in range(n):
            m = int(self.moves[i])
            make_move(self.tab, self.bb, self.occ, self.sq, self.st, self.undo, m)
            if is_legal_after_make(self.tab, self.bb, self.occ, self.st):
                legal.append(m)
            unmake_move(self.bb, self.occ, self.sq, self.st, self.undo)
        return legal

    def push(self, m: int) -> None:
        make_move(self.tab, self.bb, self.occ, self.sq, self.st, self.undo, m)

    def pop(self) -> None:
        unmake_move(self.bb, self.occ, self.sq, self.st, self.undo)

    def perft(self, depth: int) -> int:
        return int(
            perft(self.tab, self.bb, self.occ, self.sq, self.st, self.undo, self.moves, depth)
        )


def move_to_uci(m: int) -> str:
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    s = chess.SQUARE_NAMES[frm] + chess.SQUARE_NAMES[to]
    if promo:
        s += "  nbrq"[promo]
    return s
