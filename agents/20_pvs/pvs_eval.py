"""Static evaluation for 20_pvs.

Score = tapered (midgame, endgame) blend of
    incremental material + piece-square tables (kept in st[ST_MG]/st[ST_EG] by the board)
  + pawn structure (passed, isolated, doubled)
  + bishop pair
  + rooks on open / semi-open files, rook on the seventh
  + king pawn shelter (midgame)
  + piece mobility
  + mop-up term for converting won endgames
  + tempo
returned from the side to move's point of view. Insufficient-material positions score 0.

Extra lookup masks live in the flat ``ETAB`` array (same reason as the board's ``tab``:
numba embeds global arrays as constants, which is slow to compile).
"""

from __future__ import annotations

import math

import numpy as np
from numba import int64, njit

from pvs_board import (
    BISHOP,
    KING,
    KNIGHT,
    PAWN,
    PHASE_MAX,
    QUEEN,
    ROOK,
    ST_EG,
    ST_MG,
    ST_PHASE,
    ST_SIDE,
    T_FILE,
    T_KING,
    T_KNIGHT,
    WHITE,
    A1,
    bishop_attacks,
    lsb,
    popcount,
    rook_attacks,
)

# ---------------------------------------------------------------------------- masks


def _passed_masks() -> np.ndarray:
    out = np.zeros(128, dtype=np.uint64)
    for side in (0, 1):
        for s in range(64):
            f, r = s % 8, s // 8
            m = 0
            for ff in (f - 1, f, f + 1):
                if 0 <= ff < 8:
                    ranks = range(r + 1, 8) if side == 0 else range(0, r)
                    for rr in ranks:
                        m |= 1 << (rr * 8 + ff)
            out[side * 64 + s] = m
    return out.view(np.int64)


def _adjacent_files() -> np.ndarray:
    out = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        m = 0
        if f > 0:
            m |= 0x0101010101010101 << (f - 1)
        if f < 7:
            m |= 0x0101010101010101 << (f + 1)
        out[f] = m
    return out.view(np.int64)


def _shield_masks() -> np.ndarray:
    """Squares one and two ranks in front of the king on its file and neighbours."""
    out = np.zeros(128, dtype=np.uint64)
    for side in (0, 1):
        for s in range(64):
            f, r = s % 8, s // 8
            m = 0
            for ff in (f - 1, f, f + 1):
                if 0 <= ff < 8:
                    for dr in (1, 2):
                        rr = r + dr if side == 0 else r - dr
                        if 0 <= rr < 8:
                            m |= 1 << (rr * 8 + ff)
            out[side * 64 + s] = m
    return out.view(np.int64)


def _center_distance() -> np.ndarray:
    out = np.zeros(64, dtype=np.int64)
    for s in range(64):
        f, r = s % 8, s // 8
        out[s] = max(abs(f - 3.5), abs(r - 3.5)) * 2 - 1  # 0..6
    return out


def _lmr_table() -> np.ndarray:
    """Late-move reduction in plies, indexed depth * 64 + move_number."""
    out = np.zeros(64 * 64, dtype=np.int64)
    for d in range(1, 64):
        for m in range(1, 64):
            out[d * 64 + m] = int(0.5 + math.log(d) * math.log(m) / 2.1)
    return out


_eparts = [
    ("E_PASSED", _passed_masks()),
    ("E_ADJ", _adjacent_files()),
    ("E_SHIELD", _shield_masks()),
    ("E_CDIST", _center_distance()),
    ("E_LMR", _lmr_table()),
]
_eoff: dict[str, int] = {}
_p = 0
for _n, _a in _eparts:
    _eoff[_n] = _p
    _p += int(_a.size)
ETAB = np.concatenate([a.astype(np.int64) for _, a in _eparts])
E_PASSED = _eoff["E_PASSED"]
E_ADJ = _eoff["E_ADJ"]
E_SHIELD = _eoff["E_SHIELD"]
E_CDIST = _eoff["E_CDIST"]
E_LMR = _eoff["E_LMR"]

# ------------------------------------------------------------------------- weights
# (midgame, endgame) pairs in centipawns

PASSED_MG = np.array([0, 4, 8, 14, 26, 45, 75, 0], dtype=np.int64)
PASSED_EG = np.array([0, 10, 18, 32, 58, 95, 140, 0], dtype=np.int64)
SEE_VALUE = np.array([0, 100, 325, 335, 500, 975, 20000, 100, 325, 335, 500, 975, 20000],
                     dtype=np.int64)


@njit(int64(A1, A1, A1, A1, A1), cache=False)
def evaluate(tab, etab, bb, occ, st) -> int:
    all_occ = occ[2]
    wp = bb[PAWN]
    bp = bb[6 + PAWN]

    # ---- insufficient material: bare kings, or one minor piece against a bare king
    if wp == 0 and bp == 0:
        non_pawn_w = bb[KNIGHT] | bb[BISHOP] | bb[ROOK] | bb[QUEEN]
        non_pawn_b = bb[6 + KNIGHT] | bb[6 + BISHOP] | bb[6 + ROOK] | bb[6 + QUEEN]
        if (bb[ROOK] | bb[QUEEN] | bb[6 + ROOK] | bb[6 + QUEEN]) == 0:
            minors_w = popcount(non_pawn_w)
            minors_b = popcount(non_pawn_b)
            if minors_w + minors_b <= 1:
                return 0
            if minors_w == 1 and minors_b == 1:
                return 0

    mg = st[ST_MG]
    eg = st[ST_EG]

    # ---- pawn structure
    for side in range(2):
        base = 6 * side
        own = bb[base + PAWN]
        enemy = bb[(6 - base) + PAWN]
        sign = 1 if side == WHITE else -1
        pawns = own
        while pawns:
            s = lsb(pawns)
            pawns &= pawns - 1
            f = s & 7
            file_mask = tab[T_FILE + f]
            # passed
            if etab[E_PASSED + (side << 6) + s] & enemy == 0:
                rel_rank = (s >> 3) if side == WHITE else 7 - (s >> 3)
                mg += sign * PASSED_MG[rel_rank]
                eg += sign * PASSED_EG[rel_rank]
            # isolated
            if etab[E_ADJ + f] & own == 0:
                mg -= sign * 12
                eg -= sign * 18
            # doubled (count each extra pawn on the file once, from the rear pawn)
            if (file_mask & own) & ~(1 << s) and (file_mask & own) & ((1 << s) - 1) == 0:
                mg -= sign * 10
                eg -= sign * 20

    # ---- pieces: bishop pair, rooks, mobility, king
    for side in range(2):
        base = 6 * side
        sign = 1 if side == WHITE else -1
        own_occ = occ[side]
        own_pawns = bb[base + PAWN]
        enemy_pawns = bb[(6 - base) + PAWN]
        # squares attacked by enemy pawns are not counted as mobility
        if side == WHITE:
            # a black pawn attacks one rank down: >>9 lands one file left (drop file h
            # wrap-arounds), >>7 lands one file right (drop file a wrap-arounds)
            pawn_guard = ((enemy_pawns >> 9) & 0x7F7F7F7F7F7F7F7F) | (
                (enemy_pawns >> 7) & 0x00FEFEFEFEFEFEFE
            )
        else:
            pawn_guard = ((enemy_pawns << 9) & -0x0101010101010102) | (
                (enemy_pawns << 7) & 0x7F7F7F7F7F7F7F00
            )
        mob_target = ~own_occ & ~pawn_guard

        if popcount(bb[base + BISHOP]) >= 2:
            mg += sign * 30
            eg += sign * 45

        pieces = bb[base + KNIGHT]
        while pieces:
            s = lsb(pieces)
            pieces &= pieces - 1
            mob = popcount(tab[T_KNIGHT + s] & mob_target)
            mg += sign * 4 * (mob - 4)
            eg += sign * 4 * (mob - 4)

        pieces = bb[base + BISHOP]
        while pieces:
            s = lsb(pieces)
            pieces &= pieces - 1
            mob = popcount(bishop_attacks(tab, s, all_occ) & mob_target)
            mg += sign * 5 * (mob - 6)
            eg += sign * 5 * (mob - 6)

        pieces = bb[base + ROOK]
        while pieces:
            s = lsb(pieces)
            pieces &= pieces - 1
            mob = popcount(rook_attacks(tab, s, all_occ) & mob_target)
            mg += sign * 2 * (mob - 7)
            eg += sign * 4 * (mob - 7)
            file_mask = tab[T_FILE + (s & 7)]
            if file_mask & own_pawns == 0:
                if file_mask & enemy_pawns == 0:
                    mg += sign * 25
                    eg += sign * 10
                else:
                    mg += sign * 12
                    eg += sign * 5
            rel_rank = (s >> 3) if side == WHITE else 7 - (s >> 3)
            if rel_rank == 6:
                mg += sign * 10
                eg += sign * 20

        pieces = bb[base + QUEEN]
        while pieces:
            s = lsb(pieces)
            pieces &= pieces - 1
            mob = popcount(
                (rook_attacks(tab, s, all_occ) | bishop_attacks(tab, s, all_occ)) & mob_target
            )
            mg += sign * 1 * (mob - 13)
            eg += sign * 2 * (mob - 13)

        # king shelter (midgame only): own pawns in front of the king
        ks = lsb(bb[base + KING])
        shield = popcount(etab[E_SHIELD + (side << 6) + ks] & own_pawns)
        if shield > 3:
            shield = 3
        mg += sign * 12 * shield
        # open file next to the king with the enemy queen still on the board
        if bb[(6 - base) + QUEEN]:
            kf = ks & 7
            for ff in range(kf - 1, kf + 2):
                if ff < 0 or ff > 7:
                    continue
                fm = tab[T_FILE + ff]
                if fm & own_pawns == 0:
                    mg -= sign * (22 if fm & enemy_pawns == 0 else 12)

    # ---- mop-up: drive the losing king to the edge when clearly ahead with few pieces
    phase = st[ST_PHASE]
    if phase <= 8:
        wk = lsb(bb[KING])
        bk = lsb(bb[6 + KING])
        kdist = abs((wk & 7) - (bk & 7)) + abs((wk >> 3) - (bk >> 3))
        if eg > 400:
            eg += 10 * etab[E_CDIST + bk] + 4 * (14 - kdist)
        elif eg < -400:
            eg -= 10 * etab[E_CDIST + wk] + 4 * (14 - kdist)

    if phase > PHASE_MAX:
        phase = PHASE_MAX
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
    if st[ST_SIDE] == WHITE:
        return score + 12
    return -score + 12
