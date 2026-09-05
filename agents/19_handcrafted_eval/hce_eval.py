"""Handcrafted, tapered chess evaluation on bitboards.

One source file runs in two modes:

* with numba (the platform preinstalls it) every function below is compiled and the whole
  evaluation costs a few microseconds;
* without numba the same code runs as plain Python on numpy scalars, slowly but identically.

Conventions
-----------
* Squares use python-chess numbering: a1 = 0, b1 = 1, ..., h8 = 63.
* `evaluate(...)` returns centipawns from WHITE's point of view. The search negates it for Black.
* Both colours are scored by one routine, `_side_terms`, written for White. Black's pieces are
  flipped vertically before the call, so the two colours are scored by construction with the
  same code and the evaluation is exactly antisymmetric (before the tempo bonus).
* The evaluator never returns mate scores. Terminal positions belong to the search.

Feature groups (spec `19_HANDCRAFTED_EVALUATION.md`): material, piece-square tables, mobility,
pawn structure, passed pawns, king safety, bishop pair / rook files, endgame king activity and
a tapered middlegame/endgame interpolation.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

# Compiled code is cached on disk so the second process (the next arena game, or the platform
# container after validation) skips the compile. The platform allows writes only under /tmp,
# which is also where its HOME points, so the system temp dir is always a legal location.
# numba does not create a user-provided cache directory itself: with a missing directory the
# compile itself succeeds and then crashes while saving the index, which would leave the agent
# on the material-only fallback for the whole game. Create it here, and compile without a
# cache when that is impossible (or when HCE_NO_CACHE is set, which `agent.py` uses to retry).
_CACHE_DIR = os.environ.get("NUMBA_CACHE_DIR") or os.path.join(
    tempfile.gettempdir(), "hce19_numba_cache"
)
CACHE = not os.environ.get("HCE_NO_CACHE")
if CACHE:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = _CACHE_DIR
    except OSError:
        CACHE = False

try:
    if os.environ.get("HCE_NO_NUMBA"):
        raise ImportError("numba disabled by HCE_NO_NUMBA")
    from numba import boolean, int64, njit, uint64
    from numba.types import UniTuple  # type: ignore[attr-defined]

    USING_NUMBA = True
except ImportError:  # pragma: no cover - exercised by the HCE_NO_NUMBA test run

    def njit(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(function: Any) -> Any:
            return function

        return wrap

    USING_NUMBA = False
    boolean = int64 = uint64 = None  # type: ignore[assignment]

    def UniTuple(*args: Any) -> Any:  # noqa: N802
        return None

U64 = np.uint64
I64 = np.int64

# Explicit signatures: every bitboard is a uint64. Without them numba would type a small Python
# int as int64 and promote int64 op uint64 to float64. They also compile everything at import.
SIG_I_U = int64(uint64) if USING_NUMBA else None
SIG_U_U = uint64(uint64) if USING_NUMBA else None
SIG_ATT = uint64(int64, uint64) if USING_NUMBA else None
SIG_SLIDER = uint64(int64, uint64, int64, int64) if USING_NUMBA else None
SIG_I_II = int64(int64, int64) if USING_NUMBA else None
SIG_SIDE = UniTuple(int64, 2)(*([uint64] * 12)) if USING_NUMBA else None
SIG_EVAL = int64(*([uint64] * 8 + [boolean])) if USING_NUMBA else None

from hce_tables import (  # noqa: E402, F401
    BACKWARD_EG,
    BACKWARD_MG,
    BISHOP_EG,
    BISHOP_MG,
    BISHOP_OUTPOST_EG,
    BISHOP_OUTPOST_MG,
    BISHOP_PAIR_EG,
    BISHOP_PAIR_MG,
    DOUBLED_EG,
    DOUBLED_MG,
    ISOLATED_EG,
    ISOLATED_MG,
    KING_ATTACK_MAX,
    KING_ATTACK_WEIGHT,
    KING_EG,
    KING_MG,
    KNIGHT_EG,
    KNIGHT_MG,
    KNIGHT_OUTPOST_EG,
    KNIGHT_OUTPOST_MG,
    MATERIAL_EG,
    MATERIAL_MG,
    MOBILITY_BASE,
    MOBILITY_EG,
    MOBILITY_MG,
    MOPUP_EDGE,
    MOPUP_KING_DIST,
    PASSED_EG,
    PASSED_KING_DIST_EG,
    PASSED_MG,
    PASSED_SUPPORTED_EG,
    PASSED_SUPPORTED_MG,
    PAWN_EG,
    PAWN_MG,
    PAWN_STORM,
    PHALANX_EG,
    PHALANX_MG,
    PHASE_TOTAL,
    PHASE_WEIGHT,
    PST_EG,
    PST_MG,
    QUEEN_EG,
    QUEEN_MG,
    ROOK_EG,
    ROOK_MG,
    ROOK_OPEN_EG,
    ROOK_OPEN_MG,
    ROOK_SEMI_EG,
    ROOK_SEMI_MG,
    ROOK_SEVENTH_EG,
    ROOK_SEVENTH_MG,
    SHIELD_ENEMY_OPEN,
    SHIELD_MISSING,
    SHIELD_RANK1,
    SHIELD_RANK2,
    SUPPORTED_EG,
    SUPPORTED_MG,
    TEMPO_MG,
)

# ---------------------------------------------------------------------------------------------
# Bitboard tables.
# ---------------------------------------------------------------------------------------------

SQUARE_BB = np.array([1 << i for i in range(64)], dtype=np.uint64)
FILE_BB = np.array([sum(1 << (r * 8 + f) for r in range(8)) for f in range(8)], dtype=np.uint64)
RANK_BB = np.array([sum(1 << (r * 8 + f) for f in range(8)) for r in range(8)], dtype=np.uint64)
NOT_FILE_A = U64((1 << 64) - 1) & ~FILE_BB[0]
NOT_FILE_H = U64((1 << 64) - 1) & ~FILE_BB[7]
ALL_BB = U64((1 << 64) - 1)
ZERO = U64(0)
ONE = U64(1)
EIGHT = U64(8)


def _bb_of(squares: list[int]) -> int:
    out = 0
    for sq in squares:
        out |= 1 << sq
    return out


def _step_squares(sq: int, dr: int, df: int) -> list[int]:
    out = []
    r, f = sq >> 3, sq & 7
    r += dr
    f += df
    while 0 <= r < 8 and 0 <= f < 8:
        out.append(r * 8 + f)
        r += dr
        f += df
    return out


# direction order: 4 diagonals then 4 orthogonals
DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
RAY_SQ = np.full((64, 8, 8), -1, dtype=np.int32)
for _sq in range(64):
    for _d, (_dr, _df) in enumerate(DIRECTIONS):
        for _i, _s in enumerate(_step_squares(_sq, _dr, _df)):
            RAY_SQ[_sq, _d, _i] = _s

KNIGHT_ATT = np.zeros(64, dtype=np.uint64)
KING_ATT = np.zeros(64, dtype=np.uint64)
for _sq in range(64):
    _r, _f = _sq >> 3, _sq & 7
    _k = []
    for _dr, _df in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
        if 0 <= _r + _dr < 8 and 0 <= _f + _df < 8:
            _k.append((_r + _dr) * 8 + _f + _df)
    KNIGHT_ATT[_sq] = _bb_of(_k)
    _kk = []
    for _dr in (-1, 0, 1):
        for _df in (-1, 0, 1):
            if (_dr or _df) and 0 <= _r + _dr < 8 and 0 <= _f + _df < 8:
                _kk.append((_r + _dr) * 8 + _f + _df)
    KING_ATT[_sq] = _bb_of(_kk)

# white-frame masks per square
PASSED_MASK = np.zeros(64, dtype=np.uint64)  # enemy pawns here stop a white pawn from passing
BEHIND_ADJ = np.zeros(64, dtype=np.uint64)  # own pawns here can support/defend later
FRONT_SPAN = np.zeros(64, dtype=np.uint64)  # squares straight ahead
OUTPOST_MASK = np.zeros(64, dtype=np.uint64)  # enemy pawns here can chase a piece away
for _sq in range(64):
    _r, _f = _sq >> 3, _sq & 7
    _passed = 0
    _behind = 0
    _front = 0
    _outpost = 0
    for _ff in (_f - 1, _f, _f + 1):
        if not 0 <= _ff < 8:
            continue
        for _rr in range(_r + 1, 8):
            _passed |= 1 << (_rr * 8 + _ff)
            if _ff != _f:
                _outpost |= 1 << (_rr * 8 + _ff)
        if _ff != _f:
            for _rr in range(0, _r + 1):
                _behind |= 1 << (_rr * 8 + _ff)
    for _rr in range(_r + 1, 8):
        _front |= 1 << (_rr * 8 + _f)
    PASSED_MASK[_sq] = _passed
    BEHIND_ADJ[_sq] = _behind
    FRONT_SPAN[_sq] = _front
    OUTPOST_MASK[_sq] = _outpost

# king zone: the king's square, its neighbours and the three squares one rank further towards
# the enemy. "US" zones extend north (white frame), "THEM" zones extend south.
KING_ZONE_US = np.zeros(64, dtype=np.uint64)
KING_ZONE_THEM = np.zeros(64, dtype=np.uint64)
for _sq in range(64):
    _base = int(KING_ATT[_sq]) | (1 << _sq)
    _north = (_base << 8) & ((1 << 64) - 1)
    _south = _base >> 8
    KING_ZONE_US[_sq] = _base | _north
    KING_ZONE_THEM[_sq] = _base | _south

CENTER_DISTANCE = np.zeros(64, dtype=np.int32)  # Chebyshev distance to the centre 4 squares
for _sq in range(64):
    _r, _f = _sq >> 3, _sq & 7
    CENTER_DISTANCE[_sq] = max(abs(2 * _r - 7), abs(2 * _f - 7)) // 2

# ---------------------------------------------------------------------------------------------
# Bit helpers.
# ---------------------------------------------------------------------------------------------


@njit(SIG_I_U, cache=CACHE)
def popcount(bb):  # type: ignore[no-untyped-def]
    n = 0
    while bb:
        bb &= bb - ONE
        n += 1
    return n


@njit(SIG_I_U, cache=CACHE)
def lsb(bb):  # type: ignore[no-untyped-def]
    """Index of the lowest set bit; bb must be non-zero."""
    n = 0
    if (bb & U64(0xFFFFFFFF)) == ZERO:
        n += 32
        bb >>= U64(32)
    if (bb & U64(0xFFFF)) == ZERO:
        n += 16
        bb >>= U64(16)
    if (bb & U64(0xFF)) == ZERO:
        n += 8
        bb >>= EIGHT
    if (bb & U64(0xF)) == ZERO:
        n += 4
        bb >>= U64(4)
    if (bb & U64(0x3)) == ZERO:
        n += 2
        bb >>= U64(2)
    if (bb & ONE) == ZERO:
        n += 1
    return n


@njit(SIG_U_U, cache=CACHE)
def flip_vertical(bb):  # type: ignore[no-untyped-def]
    out = ZERO
    for _ in range(8):
        out = (out << EIGHT) | (bb & U64(0xFF))
        bb >>= EIGHT
    return out


@njit(SIG_SLIDER, cache=CACHE)
def slider_attacks(sq, occupied, first_dir, last_dir):  # type: ignore[no-untyped-def]
    """Attack set of a sliding piece on `sq` over the directions [first_dir, last_dir)."""
    att = ZERO
    for d in range(first_dir, last_dir):
        for i in range(8):
            s = RAY_SQ[sq, d, i]
            if s < 0:
                break
            att |= SQUARE_BB[s]
            if occupied & SQUARE_BB[s]:
                break
    return att


@njit(SIG_ATT, cache=CACHE)
def bishop_attacks(sq, occupied):  # type: ignore[no-untyped-def]
    return slider_attacks(sq, occupied, 0, 4)


@njit(SIG_ATT, cache=CACHE)
def rook_attacks(sq, occupied):  # type: ignore[no-untyped-def]
    return slider_attacks(sq, occupied, 4, 8)


@njit(SIG_U_U, cache=CACHE)
def white_pawn_attacks(pawns):  # type: ignore[no-untyped-def]
    return ((pawns << U64(7)) & NOT_FILE_H) | ((pawns << U64(9)) & NOT_FILE_A)


@njit(SIG_U_U, cache=CACHE)
def black_pawn_attacks(pawns):  # type: ignore[no-untyped-def]
    return ((pawns >> U64(9)) & NOT_FILE_H) | ((pawns >> U64(7)) & NOT_FILE_A)


@njit(SIG_I_II, cache=CACHE)
def chebyshev(a, b):  # type: ignore[no-untyped-def]
    dr = (a >> 3) - (b >> 3)
    df = (a & 7) - (b & 7)
    if dr < 0:
        dr = -dr
    if df < 0:
        df = -df
    return dr if dr > df else df


# ---------------------------------------------------------------------------------------------
# One side's terms, written for White. Returns (mg, eg) for that side.
# ---------------------------------------------------------------------------------------------


@njit(SIG_SIDE, cache=CACHE)
def _side_terms(  # type: ignore[no-untyped-def]
    p, n, b, r, q, k,  # our pieces (white frame)
    ep, en, eb, er, eq, ek,  # their pieces (white frame)
):
    mg = 0
    eg = 0
    us = p | n | b | r | q | k
    them = ep | en | eb | er | eq | ek
    occupied = us | them
    our_pawn_att = white_pawn_attacks(p)
    their_pawn_att = black_pawn_attacks(ep)
    ksq = lsb(k)
    eksq = lsb(ek)
    enemy_zone = KING_ZONE_THEM[eksq]
    king_units = 0
    king_attackers = 0
    # squares a piece may go to for mobility: not our own pieces, not covered by enemy pawns
    mobility_area = ~(us | their_pawn_att)

    # ---- pawns -----------------------------------------------------------------------------
    bb = p
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        rank = sq >> 3
        file = sq & 7
        mg += MATERIAL_MG[1] + PST_MG[1, sq]
        eg += MATERIAL_EG[1] + PST_EG[1, sq]
        sq_bb = SQUARE_BB[sq]
        neighbours = BEHIND_ADJ[sq] | OUTPOST_MASK[sq]  # own pawns on adjacent files, any rank
        supported = (our_pawn_att & sq_bb) != ZERO
        phalanx = (p & RANK_BB[rank] & ((sq_bb << ONE) | (sq_bb >> ONE)) & ~FILE_BB[file]) != ZERO
        if supported:
            mg += SUPPORTED_MG
            eg += SUPPORTED_EG
        if phalanx:
            mg += PHALANX_MG
            eg += PHALANX_EG
        if (p & FRONT_SPAN[sq]) != ZERO:
            mg -= DOUBLED_MG
            eg -= DOUBLED_EG
        if (p & neighbours) == ZERO:
            mg -= ISOLATED_MG
            eg -= ISOLATED_EG
        elif (p & BEHIND_ADJ[sq]) == ZERO and rank < 7 and (
            (SQUARE_BB[sq + 8] & their_pawn_att) != ZERO
        ):
            mg -= BACKWARD_MG
            eg -= BACKWARD_EG
        if (ep & PASSED_MASK[sq]) == ZERO and (p & FRONT_SPAN[sq]) == ZERO:
            mg += PASSED_MG[rank]
            eg += PASSED_EG[rank]
            if supported or phalanx:
                mg += PASSED_SUPPORTED_MG[rank]
                eg += PASSED_SUPPORTED_EG[rank]
            if rank < 7:
                front = sq + 8
                scale = PASSED_KING_DIST_EG[rank]
                eg += scale * chebyshev(eksq, front)
                eg -= scale * chebyshev(ksq, front) // 2
                # a free path ahead is worth more in the endgame
                if (FRONT_SPAN[sq] & occupied) == ZERO:
                    eg += PASSED_EG[rank] // 4

    # ---- knights ---------------------------------------------------------------------------
    bb = n
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        mg += MATERIAL_MG[2] + PST_MG[2, sq]
        eg += MATERIAL_EG[2] + PST_EG[2, sq]
        att = KNIGHT_ATT[sq]
        mob = popcount(att & mobility_area) - MOBILITY_BASE[2]
        mg += mob * MOBILITY_MG[2]
        eg += mob * MOBILITY_EG[2]
        rank = sq >> 3
        if 3 <= rank <= 5 and (our_pawn_att & SQUARE_BB[sq]) != ZERO and (
            ep & OUTPOST_MASK[sq]
        ) == ZERO:
            mg += KNIGHT_OUTPOST_MG
            eg += KNIGHT_OUTPOST_EG
        hits = att & enemy_zone
        if hits:
            king_attackers += 1
            king_units += KING_ATTACK_WEIGHT[2] * popcount(hits)

    # ---- bishops ---------------------------------------------------------------------------
    bb = b
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        mg += MATERIAL_MG[3] + PST_MG[3, sq]
        eg += MATERIAL_EG[3] + PST_EG[3, sq]
        att = bishop_attacks(sq, occupied)
        mob = popcount(att & mobility_area) - MOBILITY_BASE[3]
        mg += mob * MOBILITY_MG[3]
        eg += mob * MOBILITY_EG[3]
        rank = sq >> 3
        if 3 <= rank <= 5 and (our_pawn_att & SQUARE_BB[sq]) != ZERO and (
            ep & OUTPOST_MASK[sq]
        ) == ZERO:
            mg += BISHOP_OUTPOST_MG
            eg += BISHOP_OUTPOST_EG
        hits = att & enemy_zone
        if hits:
            king_attackers += 1
            king_units += KING_ATTACK_WEIGHT[3] * popcount(hits)
    if popcount(b) >= 2:
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG

    # ---- rooks -----------------------------------------------------------------------------
    bb = r
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        mg += MATERIAL_MG[4] + PST_MG[4, sq]
        eg += MATERIAL_EG[4] + PST_EG[4, sq]
        att = rook_attacks(sq, occupied)
        mob = popcount(att & mobility_area) - MOBILITY_BASE[4]
        mg += mob * MOBILITY_MG[4]
        eg += mob * MOBILITY_EG[4]
        file_bb = FILE_BB[sq & 7]
        if (p & file_bb) == ZERO:
            if (ep & file_bb) == ZERO:
                mg += ROOK_OPEN_MG
                eg += ROOK_OPEN_EG
            else:
                mg += ROOK_SEMI_MG
                eg += ROOK_SEMI_EG
        if (sq >> 3) == 6 and ((eksq >> 3) == 7 or (ep & RANK_BB[6]) != ZERO):
            mg += ROOK_SEVENTH_MG
            eg += ROOK_SEVENTH_EG
        hits = att & enemy_zone
        if hits:
            king_attackers += 1
            king_units += KING_ATTACK_WEIGHT[4] * popcount(hits)

    # ---- queens ----------------------------------------------------------------------------
    bb = q
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        mg += MATERIAL_MG[5] + PST_MG[5, sq]
        eg += MATERIAL_EG[5] + PST_EG[5, sq]
        att = bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
        mob = popcount(att & mobility_area) - MOBILITY_BASE[5]
        mg += mob * MOBILITY_MG[5]
        eg += mob * MOBILITY_EG[5]
        hits = att & enemy_zone
        if hits:
            king_attackers += 1
            king_units += KING_ATTACK_WEIGHT[5] * popcount(hits)

    # ---- king ------------------------------------------------------------------------------
    mg += PST_MG[6, ksq]
    eg += PST_EG[6, ksq]

    # pawn shield and open files around our king (middlegame only)
    krank = ksq >> 3
    kfile = ksq & 7
    if krank < 6:
        for f in range(kfile - 1, kfile + 2):
            if f < 0 or f > 7:
                continue
            file_bb = FILE_BB[f]
            ahead = file_bb & ~RANK_BB[krank] & ~(RANK_BB[krank] - ONE)  # ranks above the king
            own_ahead = p & ahead
            if own_ahead == ZERO:
                mg -= SHIELD_MISSING
                if (ep & file_bb) == ZERO and (er | eq) != ZERO:
                    mg -= SHIELD_ENEMY_OPEN
            else:
                if (own_ahead & RANK_BB[krank + 1]) != ZERO:
                    mg += SHIELD_RANK1
                elif krank < 6 and (own_ahead & RANK_BB[krank + 2]) != ZERO:
                    mg += SHIELD_RANK2
            storm = ep & file_bb & (RANK_BB[krank + 1] | RANK_BB[min(krank + 2, 7)])
            if storm != ZERO:
                mg -= PAWN_STORM

    # pressure we exert on their king
    if king_units > 0:
        if king_attackers < 2:
            king_units //= 2
        if q == ZERO:
            king_units //= 2
        danger = king_units * king_units // 6
        if danger > KING_ATTACK_MAX:
            danger = KING_ATTACK_MAX
        mg += danger

    return mg, eg


@njit(SIG_EVAL, cache=CACHE)
def evaluate(pawns, knights, bishops, rooks, queens, kings, white, black, turn):  # type: ignore[no-untyped-def]
    """Centipawns from White's point of view.

    Arguments are the python-chess bitboards (`board.pawns`, ..., `board.occupied_co[WHITE]`,
    `board.occupied_co[BLACK]`) and `board.turn` (True for White to move).
    """
    wp, wn, wb, wr, wq, wk = (
        pawns & white, knights & white, bishops & white, rooks & white, queens & white,
        kings & white,
    )
    bp, bn, bb_, br, bq, bk = (
        pawns & black, knights & black, bishops & black, rooks & black, queens & black,
        kings & black,
    )
    if wk == ZERO or bk == ZERO:  # malformed position; never happens in a real game
        return 0

    phase = (
        PHASE_WEIGHT[2] * (popcount(wn) + popcount(bn))
        + PHASE_WEIGHT[3] * (popcount(wb) + popcount(bb_))
        + PHASE_WEIGHT[4] * (popcount(wr) + popcount(br))
        + PHASE_WEIGHT[5] * (popcount(wq) + popcount(bq))
    )
    if phase > PHASE_TOTAL:
        phase = PHASE_TOTAL

    mg_w, eg_w = _side_terms(wp, wn, wb, wr, wq, wk, bp, bn, bb_, br, bq, bk)
    mg_b, eg_b = _side_terms(
        flip_vertical(bp), flip_vertical(bn), flip_vertical(bb_), flip_vertical(br),
        flip_vertical(bq), flip_vertical(bk),
        flip_vertical(wp), flip_vertical(wn), flip_vertical(wb), flip_vertical(wr),
        flip_vertical(wq), flip_vertical(wk),
    )
    mg = mg_w - mg_b
    eg = eg_w - eg_b

    # ---- endgame specifics --------------------------------------------------------------
    w_nonpawn = (
        MATERIAL_EG[2] * popcount(wn) + MATERIAL_EG[3] * popcount(wb)
        + MATERIAL_EG[4] * popcount(wr) + MATERIAL_EG[5] * popcount(wq)
    )
    b_nonpawn = (
        MATERIAL_EG[2] * popcount(bn) + MATERIAL_EG[3] * popcount(bb_)
        + MATERIAL_EG[4] * popcount(br) + MATERIAL_EG[5] * popcount(bq)
    )
    wksq = lsb(wk)
    bksq = lsb(bk)
    if bp == ZERO and w_nonpawn - b_nonpawn >= MATERIAL_EG[4] - 50:
        # White is mopping up: push the black king to the edge, bring the white king close
        eg += MOPUP_EDGE * CENTER_DISTANCE[bksq] + MOPUP_KING_DIST * (7 - chebyshev(wksq, bksq))
    elif wp == ZERO and b_nonpawn - w_nonpawn >= MATERIAL_EG[4] - 50:
        eg -= MOPUP_EDGE * CENTER_DISTANCE[wksq] + MOPUP_KING_DIST * (7 - chebyshev(wksq, bksq))

    if turn:
        mg += TEMPO_MG
    else:
        mg -= TEMPO_MG

    score = int((mg * phase + eg * (PHASE_TOTAL - phase)) / PHASE_TOTAL)

    # ---- drawishness scaling -------------------------------------------------------------
    # a lone minor piece cannot win; scale a pawnless advantage of at most one minor piece to 0
    if (score > 0 and wp == ZERO and w_nonpawn <= MATERIAL_EG[3]) or (
        score < 0 and bp == ZERO and b_nonpawn <= MATERIAL_EG[3]
    ):
        score = 0
    # opposite-coloured bishops with only pawns otherwise: halve the score
    elif (
        wn == ZERO and bn == ZERO and wr == ZERO and br == ZERO and wq == ZERO and bq == ZERO
        and popcount(wb) == 1 and popcount(bb_) == 1
    ):
        light = U64(0x55AA55AA55AA55AA)
        if ((wb & light) == ZERO) != ((bb_ & light) == ZERO):
            score = int(score / 2)
    return score


def evaluate_board(board) -> int:  # type: ignore[no-untyped-def]
    """White-relative score for a python-chess board."""
    return int(
        evaluate(
            board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
            board.occupied_co[True], board.occupied_co[False], board.turn,
        )
    )


def evaluate_stm(board) -> int:  # type: ignore[no-untyped-def]
    """Score from the side to move's point of view, what a negamax search consumes."""
    score = evaluate(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[True], board.occupied_co[False], board.turn,
    )
    return int(score) if board.turn else -int(score)


def warm_up() -> None:
    """Compile every jitted function with the argument types the search really uses."""
    import chess

    for fen in (
        chess.STARTING_FEN,
        "r3k2r/pp1n1ppp/2pbpq2/3p4/3P1B2/2NBPN2/PPQ2PPP/R3K2R w KQkq - 0 10",
        "8/5k2/8/8/8/8/3K4/4R3 w - - 0 1",
        "8/8/8/8/8/8/6k1/4K2r b - - 0 1",
    ):
        evaluate_board(chess.Board(fen))


warm_up()
