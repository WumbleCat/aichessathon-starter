"""Evaluation weights and piece-square tables shared by the compiled evaluation
(`hce_eval.py`) and the simple fallback (`hce_eval_simple.py`).

Pure numpy, no numba, so importing this file is instant.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------
# Weights. Every number is a plain integer in centipawns; "_MG" applies in the middlegame,
# "_EG" in the endgame, the phase blends them.
# ---------------------------------------------------------------------------------------------

# index: 0 unused, 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king (python-chess order)
MATERIAL_MG = np.array([0, 95, 325, 340, 500, 980, 0], dtype=np.int32)
MATERIAL_EG = np.array([0, 118, 318, 345, 550, 1010, 0], dtype=np.int32)
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
PHASE_TOTAL = 24

BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 48
TEMPO_MG = 12

# mobility: (count - baseline) * weight, per piece type
MOBILITY_MG = np.array([0, 0, 5, 5, 2, 1, 0], dtype=np.int32)
MOBILITY_EG = np.array([0, 0, 4, 5, 4, 3, 0], dtype=np.int32)
MOBILITY_BASE = np.array([0, 0, 4, 6, 7, 13, 0], dtype=np.int32)

DOUBLED_MG, DOUBLED_EG = 10, 22
ISOLATED_MG, ISOLATED_EG = 12, 16
BACKWARD_MG, BACKWARD_EG = 8, 10
SUPPORTED_MG, SUPPORTED_EG = 6, 4
PHALANX_MG, PHALANX_EG = 4, 3

# indexed by rank (0 = rank 1) of a passed pawn
PASSED_MG = np.array([0, 4, 8, 16, 32, 60, 100, 0], dtype=np.int32)
PASSED_EG = np.array([0, 10, 18, 32, 60, 100, 160, 0], dtype=np.int32)
PASSED_SUPPORTED_MG = np.array([0, 0, 4, 8, 15, 25, 40, 0], dtype=np.int32)
PASSED_SUPPORTED_EG = np.array([0, 0, 5, 10, 20, 35, 55, 0], dtype=np.int32)
PASSED_KING_DIST_EG = np.array([0, 0, 2, 4, 6, 9, 12, 0], dtype=np.int32)

ROOK_OPEN_MG, ROOK_OPEN_EG = 26, 14
ROOK_SEMI_MG, ROOK_SEMI_EG = 12, 8
ROOK_SEVENTH_MG, ROOK_SEVENTH_EG = 18, 30
KNIGHT_OUTPOST_MG, KNIGHT_OUTPOST_EG = 26, 14
BISHOP_OUTPOST_MG, BISHOP_OUTPOST_EG = 14, 6

SHIELD_RANK1 = 14  # own pawn directly in front of the king
SHIELD_RANK2 = 7  # own pawn two ranks in front
SHIELD_MISSING = 14  # no own pawn ahead of the king on that file
SHIELD_ENEMY_OPEN = 12  # ...and no enemy pawn either, with enemy rooks or queen alive
PAWN_STORM = 8  # enemy pawn within two ranks of the king

KING_ATTACK_WEIGHT = np.array([0, 0, 2, 2, 3, 5, 0], dtype=np.int32)
KING_ATTACK_MAX = 500

# mop-up: drive the defending king to the edge, bring our king close
MOPUP_EDGE = 10
MOPUP_KING_DIST = 6


# ---------------------------------------------------------------------------------------------
# Piece-square tables, written from White's side with rank 8 on the first line, then reordered
# to python-chess square numbering at import time.
# ---------------------------------------------------------------------------------------------

def _visual(table: list[int]) -> np.ndarray:
    assert len(table) == 64
    out = np.zeros(64, dtype=np.int32)
    for row in range(8):
        for col in range(8):
            out[(7 - row) * 8 + col] = table[row * 8 + col]
    return out


PAWN_MG = _visual([
    0, 0, 0, 0, 0, 0, 0, 0,
    45, 50, 50, 55, 55, 50, 50, 45,
    10, 12, 22, 32, 32, 22, 12, 10,
    4, 6, 12, 26, 26, 12, 6, 4,
    0, 0, 2, 20, 20, 2, 0, 0,
    4, -4, -8, 2, 2, -8, -4, 4,
    4, 10, 10, -22, -22, 10, 10, 4,
    0, 0, 0, 0, 0, 0, 0, 0,
])
PAWN_EG = _visual([
    0, 0, 0, 0, 0, 0, 0, 0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    28, 28, 28, 28, 28, 28, 28, 28,
    14, 14, 14, 14, 14, 14, 14, 14,
    4, 4, 4, 4, 4, 4, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
])
KNIGHT_MG = _visual([
    -55, -40, -30, -30, -30, -30, -40, -55,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 12, 16, 16, 12, 5, -30,
    -30, 4, 16, 22, 22, 16, 4, -30,
    -30, 4, 16, 22, 22, 16, 4, -30,
    -30, 4, 12, 16, 16, 12, 4, -30,
    -40, -20, 0, 6, 6, 0, -20, -40,
    -55, -30, -30, -30, -30, -30, -30, -55,
])
KNIGHT_EG = _visual([
    -45, -35, -25, -25, -25, -25, -35, -45,
    -35, -15, 0, 4, 4, 0, -15, -35,
    -25, 4, 10, 14, 14, 10, 4, -25,
    -25, 4, 14, 18, 18, 14, 4, -25,
    -25, 4, 14, 18, 18, 14, 4, -25,
    -25, 4, 10, 14, 14, 10, 4, -25,
    -35, -15, 0, 4, 4, 0, -15, -35,
    -45, -35, -25, -25, -25, -25, -35, -45,
])
BISHOP_MG = _visual([
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 8, 0, 0, 0, 0, 8, -10,
    -20, -10, -14, -10, -10, -14, -10, -20,
])
BISHOP_EG = _visual([
    -14, -8, -8, -8, -8, -8, -8, -14,
    -8, 0, 0, 0, 0, 0, 0, -8,
    -8, 0, 4, 8, 8, 4, 0, -8,
    -8, 4, 4, 10, 10, 4, 4, -8,
    -8, 0, 8, 10, 10, 8, 0, -8,
    -8, 4, 4, 8, 8, 4, 4, -8,
    -8, 0, 0, 0, 0, 0, 0, -8,
    -14, -8, -8, -8, -8, -8, -8, -14,
])
ROOK_MG = _visual([
    0, 0, 0, 0, 0, 0, 0, 0,
    8, 12, 12, 12, 12, 12, 12, 8,
    -4, 0, 0, 0, 0, 0, 0, -4,
    -4, 0, 0, 0, 0, 0, 0, -4,
    -4, 0, 0, 0, 0, 0, 0, -4,
    -4, 0, 0, 0, 0, 0, 0, -4,
    -4, 0, 0, 0, 0, 0, 0, -4,
    -2, -4, 2, 8, 8, 2, -4, -2,
])
ROOK_EG = _visual([
    2, 2, 2, 2, 2, 2, 2, 2,
    4, 6, 6, 6, 6, 6, 6, 4,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
])
QUEEN_MG = _visual([
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 2, 5, 5, 5, 5, 2, -10,
    -10, 0, 2, 0, 0, 2, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
])
QUEEN_EG = _visual([
    -18, -10, -8, -6, -6, -8, -10, -18,
    -10, -4, 0, 2, 2, 0, -4, -10,
    -8, 0, 6, 8, 8, 6, 0, -8,
    -6, 2, 8, 12, 12, 8, 2, -6,
    -6, 2, 8, 12, 12, 8, 2, -6,
    -8, 0, 6, 8, 8, 6, 0, -8,
    -10, -4, 0, 2, 2, 0, -4, -10,
    -18, -10, -8, -6, -6, -8, -10, -18,
])
KING_MG = _visual([
    -40, -50, -50, -60, -60, -50, -50, -40,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -30, -30, -30, -30, -20,
    10, 10, -5, -15, -15, -5, 10, 10,
    20, 35, 12, -10, 0, -6, 38, 22,
])
KING_EG = _visual([
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
])

PST_MG = np.zeros((7, 64), dtype=np.int32)
PST_EG = np.zeros((7, 64), dtype=np.int32)
for _pt, (_mg, _eg) in enumerate(
    [(PAWN_MG, PAWN_EG), (KNIGHT_MG, KNIGHT_EG), (BISHOP_MG, BISHOP_EG),
     (ROOK_MG, ROOK_EG), (QUEEN_MG, QUEEN_EG), (KING_MG, KING_EG)],
    start=1,
):
    PST_MG[_pt] = _mg
    PST_EG[_pt] = _eg

