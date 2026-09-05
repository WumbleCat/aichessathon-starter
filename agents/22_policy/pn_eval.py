"""Hand-written tapered evaluation: material + piece-square tables + a few cheap terms.

All numbers here were chosen by hand for this project. Scores are in centipawns from the
side-to-move point of view (positive = good for the side to move).
"""

from __future__ import annotations

import chess

MATE = 100_000
MATE_BOUND = MATE - 1_000  # scores beyond this are mates

_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
MG_VALUE = dict(zip(_PIECES, (82, 337, 365, 477, 1025, 0), strict=True))
EG_VALUE = dict(zip(_PIECES, (94, 281, 297, 512, 936, 0), strict=True))
PHASE_WEIGHT = dict(zip(_PIECES, (0, 1, 1, 2, 4, 0), strict=True))
TOTAL_PHASE = 24

# Tables are written from WHITE's point of view with rank 8 on the first row (as a diagram),
# so they read naturally; they are flipped into square order below.
_PAWN_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    60, 70, 60, 60, 60, 60, 70, 60,
    20, 30, 35, 45, 45, 35, 30, 20,
    5, 10, 15, 30, 30, 15, 10, 5,
    0, 5, 10, 25, 25, 10, 5, 0,
    2, 0, 5, 5, 5, 5, 0, 2,
    5, 10, 5, -20, -20, 5, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PAWN_EG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    120, 115, 110, 100, 100, 110, 115, 120,
    70, 70, 60, 50, 50, 60, 70, 70,
    30, 28, 22, 18, 18, 22, 28, 30,
    12, 10, 6, 4, 4, 6, 10, 12,
    4, 4, 0, 0, 0, 0, 4, 4,
    6, 6, 4, 4, 4, 4, 6, 6,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT_MG = [
    -60, -40, -30, -30, -30, -30, -40, -60,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 5, 20, 25, 25, 20, 5, -30,
    -30, 0, 20, 25, 25, 20, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -60, -30, -30, -30, -30, -30, -30, -60,
]
_KNIGHT_EG = [
    -50, -35, -25, -20, -20, -25, -35, -50,
    -35, -20, -5, 0, 0, -5, -20, -35,
    -25, -5, 10, 15, 15, 10, -5, -25,
    -20, 0, 15, 20, 20, 15, 0, -20,
    -20, 0, 15, 20, 20, 15, 0, -20,
    -25, -5, 10, 15, 15, 10, -5, -25,
    -35, -20, -5, 0, 0, -5, -20, -35,
    -50, -35, -25, -20, -20, -25, -35, -50,
]
_BISHOP_MG = [
    -25, -10, -10, -10, -10, -10, -10, -25,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 15, 15, 10, 0, -10,
    -10, 5, 5, 15, 15, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 0, 0, 0, 0, 10, -10,
    -25, -10, -15, -10, -10, -15, -10, -25,
]
_BISHOP_EG = [
    -15, -10, -10, -10, -10, -10, -10, -15,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 10, 15, 15, 10, 0, -10,
    -10, 0, 10, 15, 15, 10, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -15, -10, -10, -10, -10, -10, -10, -15,
]
_ROOK_MG = [
    10, 10, 15, 20, 20, 15, 10, 10,
    20, 25, 25, 30, 30, 25, 25, 20,
    0, 5, 5, 10, 10, 5, 5, 0,
    -5, 0, 0, 5, 5, 0, 0, -5,
    -10, -5, 0, 5, 5, 0, -5, -10,
    -15, -5, 0, 0, 0, 0, -5, -15,
    -20, -10, -5, 0, 0, -5, -10, -20,
    -15, -10, 0, 10, 10, 0, -10, -15,
]
_ROOK_EG = [
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 12, 12, 12, 12, 12, 12, 10,
    5, 8, 8, 8, 8, 8, 8, 5,
    2, 4, 4, 4, 4, 4, 4, 2,
    0, 0, 0, 0, 0, 0, 0, 0,
    -4, -2, -2, -2, -2, -2, -2, -4,
    -6, -4, -4, -4, -4, -4, -4, -6,
    -6, -4, -2, 0, 0, -2, -4, -6,
]
_QUEEN_MG = [
    -25, -10, -5, 0, 0, -5, -10, -25,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 5, 5, 8, 8, 5, 5, -10,
    -5, 0, 8, 10, 10, 8, 0, -5,
    -5, 0, 8, 10, 10, 8, 0, -5,
    -10, 5, 5, 8, 8, 5, 5, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -25, -10, -10, 5, -5, -10, -10, -25,
]
_QUEEN_EG = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 15, 20, 20, 15, 5, -10,
    -5, 10, 20, 25, 25, 20, 10, -5,
    -5, 10, 20, 25, 25, 20, 10, -5,
    -10, 5, 15, 20, 20, 15, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_MG = [
    -70, -70, -70, -80, -80, -70, -70, -70,
    -60, -60, -60, -70, -70, -60, -60, -60,
    -50, -50, -55, -60, -60, -55, -50, -50,
    -40, -45, -50, -60, -60, -50, -45, -40,
    -30, -35, -40, -50, -50, -40, -35, -30,
    -20, -25, -30, -30, -30, -30, -25, -20,
    5, 5, -5, -20, -20, -5, 5, 5,
    15, 30, 10, -10, 0, -10, 30, 15,
]
_KING_EG = [
    -60, -40, -30, -20, -20, -30, -40, -60,
    -30, -15, -5, 0, 0, -5, -15, -30,
    -20, 0, 15, 20, 20, 15, 0, -20,
    -20, 0, 20, 30, 30, 20, 0, -20,
    -20, 0, 20, 30, 30, 20, 0, -20,
    -20, 0, 15, 20, 20, 15, 0, -20,
    -30, -15, 0, 5, 5, 0, -15, -30,
    -60, -40, -30, -20, -20, -30, -40, -60,
]

_DIAGRAMS = {
    chess.PAWN: (_PAWN_MG, _PAWN_EG),
    chess.KNIGHT: (_KNIGHT_MG, _KNIGHT_EG),
    chess.BISHOP: (_BISHOP_MG, _BISHOP_EG),
    chess.ROOK: (_ROOK_MG, _ROOK_EG),
    chess.QUEEN: (_QUEEN_MG, _QUEEN_EG),
    chess.KING: (_KING_MG, _KING_EG),
}


def _to_square_order(diagram: list[int]) -> list[int]:
    """Diagram row 0 is rank 8; square 0 is a1. Return a 64-list indexed by chess square."""
    out = [0] * 64
    for row in range(8):
        for file in range(8):
            rank = 7 - row
            out[chess.square(file, rank)] = diagram[row * 8 + file]
    return out


# MG[color][piece_type][square] including material; black tables are mirrored.
MG: list[list[list[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
EG: list[list[list[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
for _pt, (_mg_diag, _eg_diag) in _DIAGRAMS.items():
    _mg = _to_square_order(_mg_diag)
    _eg = _to_square_order(_eg_diag)
    for _sq in range(64):
        MG[chess.WHITE][_pt][_sq] = MG_VALUE[_pt] + _mg[_sq]
        EG[chess.WHITE][_pt][_sq] = EG_VALUE[_pt] + _eg[_sq]
        _msq = chess.square_mirror(_sq)
        MG[chess.BLACK][_pt][_msq] = MG_VALUE[_pt] + _mg[_sq]
        EG[chess.BLACK][_pt][_msq] = EG_VALUE[_pt] + _eg[_sq]

_MG_W = MG[chess.WHITE]
_MG_B = MG[chess.BLACK]
_EG_W = EG[chess.WHITE]
_EG_B = EG[chess.BLACK]

BISHOP_PAIR_MG = 25
BISHOP_PAIR_EG = 45
TEMPO = 10
PASSED_MG = [0, 4, 8, 14, 26, 45, 75, 0]
PASSED_EG = [0, 10, 16, 28, 50, 85, 130, 0]
DOUBLED = 12
ISOLATED = 10

_FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]
_ADJ_FILE_MASKS = [
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
]
# squares in front of a white pawn on `sq` (same and adjacent files, higher ranks)
_FRONT_SPAN_W = [0] * 64
_FRONT_SPAN_B = [0] * 64
for _sq in range(64):
    _f = chess.square_file(_sq)
    _r = chess.square_rank(_sq)
    _files = _FILE_MASKS[_f] | _ADJ_FILE_MASKS[_f]
    _ahead_w = 0
    _ahead_b = 0
    for _rr in range(_r + 1, 8):
        _ahead_w |= chess.BB_RANKS[_rr]
    for _rr in range(0, _r):
        _ahead_b |= chess.BB_RANKS[_rr]
    _FRONT_SPAN_W[_sq] = _files & _ahead_w
    _FRONT_SPAN_B[_sq] = _files & _ahead_b


_PAWN_CACHE: dict[tuple[int, int], tuple[int, int]] = {}


def _pawn_terms(wp: int, bp: int) -> tuple[int, int]:
    """(mg, eg) pawn material, placement and structure from White's point of view. Cached."""
    cached = _PAWN_CACHE.get((wp, bp))
    if cached is not None:
        return cached
    mg = 0
    eg = 0
    tw = _MG_W[1]
    te = _EG_W[1]
    bb = wp
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg += tw[sq]
        eg += te[sq]
        if not (_FRONT_SPAN_W[sq] & bp):
            r = sq >> 3
            mg += PASSED_MG[r]
            eg += PASSED_EG[r]
        f = sq & 7
        if not (_ADJ_FILE_MASKS[f] & wp):
            mg -= ISOLATED
            eg -= ISOLATED
    tw = _MG_B[1]
    te = _EG_B[1]
    bb = bp
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        mg -= tw[sq]
        eg -= te[sq]
        if not (_FRONT_SPAN_B[sq] & wp):
            r = 7 - (sq >> 3)
            mg -= PASSED_MG[r]
            eg -= PASSED_EG[r]
        f = sq & 7
        if not (_ADJ_FILE_MASKS[f] & bp):
            mg += ISOLATED
            eg += ISOLATED
    for fm in _FILE_MASKS:
        c = (wp & fm).bit_count()
        if c > 1:
            mg -= DOUBLED * (c - 1)
            eg -= DOUBLED * (c - 1)
        c = (bp & fm).bit_count()
        if c > 1:
            mg += DOUBLED * (c - 1)
            eg += DOUBLED * (c - 1)
    if len(_PAWN_CACHE) > 400_000:
        _PAWN_CACHE.clear()
    _PAWN_CACHE[(wp, bp)] = (mg, eg)
    return mg, eg


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns for the side to move."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pawns = board.pawns
    mg, eg = _pawn_terms(pawns & white, pawns & black)
    phase = 0
    bishops = board.bishops
    kings = board.kings

    # minor and major pieces
    for pt, pieces, weight in (
        (2, board.knights, 1),
        (3, bishops, 1),
        (4, board.rooks, 2),
        (5, board.queens, 4),
    ):
        tw = _MG_W[pt]
        te = _EG_W[pt]
        bb = pieces & white
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg += tw[sq]
            eg += te[sq]
            phase += weight
        tw = _MG_B[pt]
        te = _EG_B[pt]
        bb = pieces & black
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg -= tw[sq]
            eg -= te[sq]
            phase += weight

    wk = (kings & white).bit_length() - 1
    bk = (kings & black).bit_length() - 1
    mg += _MG_W[6][wk] - _MG_B[6][bk]
    eg += _EG_W[6][wk] - _EG_B[6][bk]

    if (bishops & white).bit_count() >= 2:
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG
    if (bishops & black).bit_count() >= 2:
        mg -= BISHOP_PAIR_MG
        eg -= BISHOP_PAIR_EG

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if board.turn == chess.BLACK:
        score = -score
    return score + TEMPO


# Plain piece values for move ordering / SEE-style pruning
PIECE_VALUE = [0, 100, 320, 330, 500, 900, 20_000]
