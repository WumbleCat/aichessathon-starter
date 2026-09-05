"""Fallback evaluation: tapered material + piece-square tables in plain Python.

Used only while the compiled evaluation in `hce_eval.py` is still being built (or if numba
is unavailable). Same tables, same sign convention (side to move), no other terms.
"""

from __future__ import annotations

import chess
from hce_tables import MATERIAL_EG, MATERIAL_MG, PHASE_TOTAL, PHASE_WEIGHT, PST_EG, PST_MG, TEMPO_MG

_MG = [[int(v) + int(MATERIAL_MG[pt]) for v in PST_MG[pt]] for pt in range(7)]
_EG = [[int(v) + int(MATERIAL_EG[pt]) for v in PST_EG[pt]] for pt in range(7)]
_PHASE = [int(v) for v in PHASE_WEIGHT]


def evaluate_stm(board: chess.Board) -> int:
    """Centipawns from the side to move's point of view."""
    mg = 0
    eg = 0
    phase = 0
    for square, piece in board.piece_map().items():
        pt = piece.piece_type
        phase += _PHASE[pt]
        if piece.color:
            mg += _MG[pt][square]
            eg += _EG[pt][square]
        else:
            mg -= _MG[pt][square ^ 56]
            eg -= _EG[pt][square ^ 56]
    phase = min(phase, PHASE_TOTAL)
    mg += TEMPO_MG if board.turn else -TEMPO_MG
    score = int((mg * phase + eg * (PHASE_TOTAL - phase)) / PHASE_TOTAL)
    return score if board.turn else -score
