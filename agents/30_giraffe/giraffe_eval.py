"""Giraffe-style feature extraction and evaluation networks, compiled with numba.

Everything here works on raw bitboards taken from a ``chess.Board`` so that no Python
object is touched per square. The board is first normalised so that the side to move is
always "us" on the white side of the board (vertical flip and colour swap when black is to
move); every value returned is therefore from the side to move's point of view, which is
what a negamax search wants, and the network is colour-symmetric by construction.

Feature layout (335 floats), following the three groups of Lai's Giraffe:

* global (15): castling rights for us/them, material counts for us/them, en passant flag;
* piece-centric (32 slots x 6): presence, file, rank, lowest-valued enemy attacker,
  lowest-valued friendly defender, mobility. Slots are K, Q, R, R, B, B, N, N, P x 8 for
  each side; surplus promoted pieces still count in the other groups;
* square-centric (64 x 2): lowest-valued attacker of each square by us and by them.

Two evaluators share this representation. ``hce_eval`` is the handcrafted material +
piece-square control evaluator. ``net_eval`` returns that exact static score plus a learned
residual: the network never has to re-learn material, it is trained on what a deeper
search knows that the static score does not (see ``training/``). Both return centipawns
for the side to move.
"""

from __future__ import annotations

import math
import os
import tempfile

# numba writes its compilation cache next to the source by default; the contest filesystem
# is read-only apart from /tmp, so point the cache there before numba is imported. A cold
# start still compiles inside the 90 s init budget; a warm cache makes it near instant.
# ``GIRAFFE_NUMBA_CACHE=0`` disables caching entirely (agent.py retries with it if the cache
# directory turns out to be unusable).
CACHE = os.environ.get("GIRAFFE_NUMBA_CACHE", "1") != "0"
if CACHE:
    _cache_dir = os.environ.setdefault(
        "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "giraffe_numba_cache")
    )
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except OSError:
        CACHE = False

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

# ---------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------

ONE = np.uint64(1)
ZERO = np.uint64(0)
BYTE = np.uint64(0xFF)
EIGHT = np.uint64(8)

# piece type codes follow python-chess: 1 pawn .. 6 king
PIECE_TENTHS = np.array([0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 10.0], dtype=np.float32)
PIECE_CP = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)  # total 24 at the start

N_GLOBAL = 15
N_SLOTS = 32
SLOT_FEATS = 6
N_PIECE = N_SLOTS * SLOT_FEATS  # 192
N_SQUARE = 128
N_INPUT = N_GLOBAL + N_PIECE + N_SQUARE  # 335

# slot bases inside one side's 16 slots, indexed by piece type; capacities likewise
SLOT_BASE = np.array([0, 8, 6, 4, 2, 1, 0], dtype=np.int64)
SLOT_CAP = np.array([0, 8, 2, 2, 2, 1, 1], dtype=np.int64)
MATERIAL_NORM = np.array([1.0, 8.0, 2.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float32)

# network shape: three first-layer groups, merged, two more hidden layers, one output
H_G = 16
H_P = 128
H_S = 64
H_MERGED = H_G + H_P + H_S  # 208
H_2 = 64
H_3 = 32
OUT_SCALE = 600.0  # residual centipawns = OUT_SCALE * tanh(z), added to the static score

LAYERS = (
    ("g", H_G, N_GLOBAL),
    ("p", H_P, N_PIECE),
    ("s", H_S, N_SQUARE),
    ("h2", H_2, H_MERGED),
    ("h3", H_3, H_2),
    ("out", 1, H_3),
)


def _layout() -> dict[str, tuple[int, int, int, int]]:
    """Offsets of each layer's weight matrix and bias inside the flat weight vector."""
    offsets: dict[str, tuple[int, int, int, int]] = {}
    cursor = 0
    for name, n_out, n_in in LAYERS:
        offsets[name] = (cursor, cursor + n_out * n_in, n_out, n_in)
        cursor += n_out * n_in + n_out
    offsets["_total"] = (cursor, cursor, 0, 0)
    return offsets


LAYOUT = _layout()
N_WEIGHTS = LAYOUT["_total"][0]

OFF_WG, OFF_BG = LAYOUT["g"][0], LAYOUT["g"][1]
OFF_WP, OFF_BP = LAYOUT["p"][0], LAYOUT["p"][1]
OFF_WS, OFF_BS = LAYOUT["s"][0], LAYOUT["s"][1]
OFF_W2, OFF_B2 = LAYOUT["h2"][0], LAYOUT["h2"][1]
OFF_W3, OFF_B3 = LAYOUT["h3"][0], LAYOUT["h3"][1]
OFF_WO, OFF_BO = LAYOUT["out"][0], LAYOUT["out"][1]


def _step_tables() -> tuple[np.ndarray, np.ndarray]:
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        f, r = sq % 8, sq // 8
        for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            if 0 <= f + df < 8 and 0 <= r + dr < 8:
                knight[sq] |= np.uint64(1) << np.uint64((r + dr) * 8 + f + df)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if (df or dr) and 0 <= f + df < 8 and 0 <= r + dr < 8:
                    king[sq] |= np.uint64(1) << np.uint64((r + dr) * 8 + f + df)
    return knight, king


KNIGHT_ATT, KING_ATT = _step_tables()

# ---------------------------------------------------------------------------------------
# Handcrafted piece-square tables (own design: centre, development, pawn advance, king
# shelter in the middlegame and king centralisation in the endgame). Indexed from white's
# point of view with a1 = 0, so rank 1 is the first row.
# ---------------------------------------------------------------------------------------


def _pst() -> tuple[np.ndarray, np.ndarray]:
    mg = np.zeros((7, 64), dtype=np.int32)
    eg = np.zeros((7, 64), dtype=np.int32)
    centre = [-3, -1, 1, 2, 2, 1, -1, -3]
    for sq in range(64):
        f, r = sq % 8, sq // 8
        c = centre[f] + centre[r]
        # pawns: advance, centre files, small penalty for the wing pawns leaving early
        mg[1, sq] = 4 * c + 6 * (r - 1) + (12 if r in (3, 4) and f in (3, 4) else 0)
        eg[1, sq] = 12 * (r - 1) + 2 * centre[f]
        mg[2, sq] = 8 * c
        eg[2, sq] = 6 * c
        mg[3, sq] = 5 * c + (6 if r > 0 else -10)
        eg[3, sq] = 4 * c
        mg[4, sq] = 2 * centre[f] + (14 if r == 6 else 0)
        eg[4, sq] = 3 * centre[f] + (8 if r == 6 else 0)
        mg[5, sq] = 3 * c - (10 if r > 1 else 0)
        eg[5, sq] = 5 * c
        mg[6, sq] = -12 * r - 8 * abs(f - 3.5) + (20 if r == 0 and f in (1, 2, 6) else 0)
        eg[6, sq] = 8 * c
    return mg, eg


PST_MG, PST_EG = _pst()

# ---------------------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------------------


@njit("uint64(uint64)", cache=CACHE)
def _flip(bb: np.uint64) -> np.uint64:
    out = ZERO
    for i in range(8):
        out = (out << EIGHT) | ((bb >> np.uint64(8 * i)) & BYTE)
    return out


@njit("int64(uint64)", cache=CACHE)
def _popcount(bb: np.uint64) -> int:
    n = 0
    while bb != ZERO:
        bb &= bb - ONE
        n += 1
    return n


@njit("uint64(int64, uint64, boolean, boolean)", cache=CACHE)
def _slider(sq: int, occ: np.uint64, bishop: bool, rook: bool) -> np.uint64:
    att = ZERO
    f0 = sq & 7
    r0 = sq >> 3
    for d in range(8):
        if d < 4:
            if not bishop:
                continue
            df = 1 if (d & 1) == 0 else -1
            dr = 1 if d < 2 else -1
        else:
            if not rook:
                continue
            if d == 4:
                df, dr = 1, 0
            elif d == 5:
                df, dr = -1, 0
            elif d == 6:
                df, dr = 0, 1
            else:
                df, dr = 0, -1
        f = f0 + df
        r = r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            m = ONE << np.uint64(r * 8 + f)
            att |= m
            if occ & m:
                break
            f += df
            r += dr
    return att


@njit("UniTuple(int64, 4)(uint64)", cache=CACHE)
def _castle_bits(castling: np.uint64) -> tuple[int, int, int, int]:
    wk = 1 if castling & (ONE << np.uint64(7)) else 0
    wq = 1 if castling & ONE else 0
    bk = 1 if castling & (ONE << np.uint64(63)) else 0
    bq = 1 if castling & (ONE << np.uint64(56)) else 0
    return wk, wq, bk, bq


# ---------------------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------------------


@njit(
    "void(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, boolean, uint64, int64, "
    "float32[::1])",
    cache=CACHE,
)
def features(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    occ_w: np.uint64,
    occ_b: np.uint64,
    white_to_move: bool,
    castling: np.uint64,
    ep_square: int,
    x: np.ndarray,
) -> None:
    """Fill ``x`` (float32[N_INPUT]) with the side-to-move normalised feature vector.

    Arguments are the raw bitboards of a ``chess.Board`` (see ``_bitboards``): piece
    type boards, white/black occupancy, side to move, castling-rights bitboard and the
    en passant square (-1 = none).
    """
    wk, wq, bk, bq = _castle_bits(castling)
    if white_to_move:
        us = occ_w
        them = occ_b
        ck_us, cq_us, ck_them, cq_them = wk, wq, bk, bq
    else:
        pawns = _flip(pawns)
        knights = _flip(knights)
        bishops = _flip(bishops)
        rooks = _flip(rooks)
        queens = _flip(queens)
        kings = _flip(kings)
        us = _flip(occ_b)
        them = _flip(occ_w)
        ck_us, cq_us, ck_them, cq_them = bk, bq, wk, wq
    occ = us | them

    for i in range(N_INPUT):
        x[i] = 0.0

    # per-square scratch: lowest attacker value by us / them, mobility of the piece there
    att_us = np.zeros(64, dtype=np.float32)
    att_them = np.zeros(64, dtype=np.float32)
    mob = np.zeros(64, dtype=np.float32)
    ptype = np.zeros(64, dtype=np.int64)
    pcol = np.zeros(64, dtype=np.int64)  # 0 us, 1 them
    counts = np.zeros(14, dtype=np.int64)  # [side*7 + type]

    for sq in range(64):
        m = ONE << np.uint64(sq)
        if not (occ & m):
            continue
        if pawns & m:
            t = 1
        elif knights & m:
            t = 2
        elif bishops & m:
            t = 3
        elif rooks & m:
            t = 4
        elif queens & m:
            t = 5
        else:
            t = 6
        side = 0 if (us & m) else 1
        ptype[sq] = t
        pcol[sq] = side
        counts[side * 7 + t] += 1
        own = us if side == 0 else them
        f = sq & 7
        r = sq >> 3
        if t == 1:
            att = ZERO
            if side == 0:
                if f > 0 and r < 7:
                    att |= ONE << np.uint64(sq + 7)
                if f < 7 and r < 7:
                    att |= ONE << np.uint64(sq + 9)
                push = ZERO
                if r < 7 and not (occ & (ONE << np.uint64(sq + 8))):
                    push |= ONE << np.uint64(sq + 8)
                    if r == 1 and not (occ & (ONE << np.uint64(sq + 16))):
                        push |= ONE << np.uint64(sq + 16)
            else:
                if f > 0 and r > 0:
                    att |= ONE << np.uint64(sq - 9)
                if f < 7 and r > 0:
                    att |= ONE << np.uint64(sq - 7)
                push = ZERO
                if r > 0 and not (occ & (ONE << np.uint64(sq - 8))):
                    push |= ONE << np.uint64(sq - 8)
                    if r == 6 and not (occ & (ONE << np.uint64(sq - 16))):
                        push |= ONE << np.uint64(sq - 16)
            enemy = them if side == 0 else us
            mob[sq] = float(_popcount(push) + _popcount(att & enemy))
        elif t == 2:
            att = KNIGHT_ATT[sq]
            mob[sq] = float(_popcount(att & ~own))
        elif t == 6:
            att = KING_ATT[sq]
            mob[sq] = float(_popcount(att & ~own))
        else:
            att = _slider(sq, occ, t == 3 or t == 5, t == 4 or t == 5)
            mob[sq] = float(_popcount(att & ~own))
        val = PIECE_TENTHS[t]
        for s2 in range(64):
            if att & (ONE << np.uint64(s2)):
                if side == 0:
                    if att_us[s2] == 0.0 or val < att_us[s2]:
                        att_us[s2] = val
                elif att_them[s2] == 0.0 or val < att_them[s2]:
                    att_them[s2] = val

    # global group
    x[0] = ck_us
    x[1] = cq_us
    x[2] = ck_them
    x[3] = cq_them
    for t in range(1, 6):
        x[3 + t] = counts[t] / MATERIAL_NORM[t]
        x[8 + t] = counts[7 + t] / MATERIAL_NORM[t]
    x[14] = 1.0 if ep_square >= 0 else 0.0

    # piece-centric group: slots filled in square order
    filled = np.zeros(14, dtype=np.int64)
    for sq in range(64):
        t = ptype[sq]
        if t == 0:
            continue
        side = pcol[sq]
        k = filled[side * 7 + t]
        if k >= SLOT_CAP[t]:
            continue
        filled[side * 7 + t] = k + 1
        slot = side * 16 + SLOT_BASE[t] + k
        base = N_GLOBAL + slot * SLOT_FEATS
        x[base] = 1.0
        x[base + 1] = (sq & 7) / 7.0
        x[base + 2] = (sq >> 3) / 7.0
        if side == 0:
            x[base + 3] = att_them[sq] / 10.0
            x[base + 4] = att_us[sq] / 10.0
        else:
            x[base + 3] = att_us[sq] / 10.0
            x[base + 4] = att_them[sq] / 10.0
        x[base + 5] = mob[sq] / 14.0

    # square-centric group
    base = N_GLOBAL + N_PIECE
    for sq in range(64):
        x[base + sq] = att_us[sq] / 10.0
        x[base + 64 + sq] = att_them[sq] / 10.0


# ---------------------------------------------------------------------------------------
# Network forward pass
# ---------------------------------------------------------------------------------------


@njit(cache=CACHE, fastmath=True)
def _dense_relu(
    w: np.ndarray,
    off_w: int,
    off_b: int,
    x: np.ndarray,
    x0: int,
    n_in: int,
    out: np.ndarray,
    o0: int,
    n_out: int,
) -> None:
    for j in range(n_out):
        acc = w[off_b + j]
        row = off_w + j * n_in
        for i in range(n_in):
            acc += w[row + i] * x[x0 + i]
        out[o0 + j] = acc if acc > 0.0 else 0.0


@njit("float64(float32[::1], float32[::1])", cache=CACHE, fastmath=True)
def forward(w: np.ndarray, x: np.ndarray) -> float:
    """Network output (the residual, in centipawns) for the feature vector ``x``."""
    h1 = np.empty(H_MERGED, dtype=np.float32)
    _dense_relu(w, OFF_WG, OFF_BG, x, 0, N_GLOBAL, h1, 0, H_G)
    _dense_relu(w, OFF_WP, OFF_BP, x, N_GLOBAL, N_PIECE, h1, H_G, H_P)
    _dense_relu(w, OFF_WS, OFF_BS, x, N_GLOBAL + N_PIECE, N_SQUARE, h1, H_G + H_P, H_S)
    h2 = np.empty(H_2, dtype=np.float32)
    _dense_relu(w, OFF_W2, OFF_B2, h1, 0, H_MERGED, h2, 0, H_2)
    h3 = np.empty(H_3, dtype=np.float32)
    _dense_relu(w, OFF_W3, OFF_B3, h2, 0, H_2, h3, 0, H_3)
    z = w[OFF_BO]
    for i in range(H_3):
        z += w[OFF_WO + i] * h3[i]
    return OUT_SCALE * math.tanh(z)


# ---------------------------------------------------------------------------------------
# Handcrafted control evaluator (material + tapered piece-square tables + bishop pair)
# ---------------------------------------------------------------------------------------


@njit("int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, boolean)", cache=CACHE)
def hce_eval_bb(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    occ_w: np.uint64,
    occ_b: np.uint64,
    white_to_move: bool,
) -> int:
    mg = 0
    eg = 0
    phase = 0
    bishops_w = 0
    bishops_b = 0
    for sq in range(64):
        m = ONE << np.uint64(sq)
        if not ((occ_w | occ_b) & m):
            continue
        if pawns & m:
            t = 1
        elif knights & m:
            t = 2
        elif bishops & m:
            t = 3
        elif rooks & m:
            t = 4
        elif queens & m:
            t = 5
        else:
            t = 6
        phase += PHASE_WEIGHT[t]
        if occ_w & m:
            mg += PIECE_CP[t] + PST_MG[t, sq]
            eg += PIECE_CP[t] + PST_EG[t, sq]
            if t == 3:
                bishops_w += 1
        else:
            fsq = sq ^ 56
            mg -= PIECE_CP[t] + PST_MG[t, fsq]
            eg -= PIECE_CP[t] + PST_EG[t, fsq]
            if t == 3:
                bishops_b += 1
    if bishops_w >= 2:
        mg += 25
        eg += 40
    if bishops_b >= 2:
        mg -= 25
        eg -= 40
    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24
    return score if white_to_move else -score


@njit(
    "float64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, boolean, uint64, "
    "int64, float32[::1], float32[::1])",
    cache=CACHE,
)
def net_eval_bb(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    occ_w: np.uint64,
    occ_b: np.uint64,
    white_to_move: bool,
    castling: np.uint64,
    ep_square: int,
    w: np.ndarray,
    scratch: np.ndarray,
) -> float:
    """Static handcrafted score plus the network residual, centipawns for the side to move."""
    features(
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        occ_w,
        occ_b,
        white_to_move,
        castling,
        ep_square,
        scratch,
    )
    static = hce_eval_bb(pawns, knights, bishops, rooks, queens, kings, occ_w, occ_b, white_to_move)
    return float(static) + forward(w, scratch)


# ---------------------------------------------------------------------------------------
# Python-facing wrappers
# ---------------------------------------------------------------------------------------


Bitboards = tuple[Any, ...]  # numba checks the real types at the call


def _bitboards(board: Any) -> Bitboards:
    """The raw bitboards of a ``chess.Board`` in the order every jitted function takes."""
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[True],
        board.occupied_co[False],
        board.turn,
        board.castling_rights,
        -1 if board.ep_square is None else board.ep_square,
    )


def board_features(board: Any) -> np.ndarray:
    """Feature vector of a ``chess.Board`` (side-to-move normalised)."""
    x = np.zeros(N_INPUT, dtype=np.float32)
    features(*_bitboards(board), x)  # type: ignore[call-arg]
    return x


class NetEvaluator:
    """Callable ``board -> centipawns`` backed by a flat weight vector."""

    def __init__(self, weights: np.ndarray) -> None:
        if weights.shape != (N_WEIGHTS,) or weights.dtype != np.float32:
            raise ValueError(f"expected float32[{N_WEIGHTS}], got {weights.dtype}{weights.shape}")
        self.weights = np.ascontiguousarray(weights)
        self.scratch = np.zeros(N_INPUT, dtype=np.float32)

    def __call__(self, board: Any) -> int:
        p, n, b, r, q, k, ow, ob, turn, castling, ep = _bitboards(board)
        return int(
            net_eval_bb(p, n, b, r, q, k, ow, ob, turn, castling, ep, self.weights, self.scratch)
        )

    def residual(self, board: Any) -> float:
        """The learned correction alone (what ``training/`` fits), in centipawns."""
        features(*_bitboards(board), self.scratch)  # type: ignore[call-arg]
        return float(forward(self.weights, self.scratch))


def hce_eval(board: Any) -> int:
    """Handcrafted evaluation in centipawns for the side to move."""
    p, n, b, r, q, k, ow, ob, turn, _castling, _ep = _bitboards(board)
    return int(hce_eval_bb(p, n, b, r, q, k, ow, ob, turn))


def random_weights(seed: int = 0) -> np.ndarray:
    """He-initialised flat weight vector, for tests and untrained smoke runs."""
    rng = np.random.default_rng(seed)
    w = np.zeros(N_WEIGHTS, dtype=np.float32)
    for name, n_out, n_in in LAYERS:
        off_w, off_b = LAYOUT[name][0], LAYOUT[name][1]
        w[off_w : off_w + n_out * n_in] = rng.normal(0.0, math.sqrt(2.0 / n_in), n_out * n_in)
        w[off_b : off_b + n_out] = 0.0
    return w


def warm_up() -> None:
    """Compile every jitted function with the argument types the search will pass."""
    import chess

    board = chess.Board()
    hce_eval(board)
    NetEvaluator(random_weights())(board)
    board.push_san("e4")
    hce_eval(board)
    board_features(board)
