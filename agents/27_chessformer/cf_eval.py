"""Fast tapered material + piece-square evaluation for the leaf nodes of cf_search.

Tables are generated from simple formulas (centralisation, advancement, king shelter) rather than
copied from any engine. Scores are in centipawns from the side to move's point of view.
"""

import chess

MATE = 100_000
MATE_BOUND = MATE - 1000  # scores beyond this are mate scores

PIECE_VALUE_MG = [0, 82, 337, 365, 477, 1025, 0]
PIECE_VALUE_EG = [0, 94, 281, 297, 512, 936, 0]
PHASE_WEIGHT = [0, 0, 1, 1, 2, 4, 0]
MAX_PHASE = 24


def _center_distance(sq: int) -> float:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return max(abs(f - 3.5), abs(r - 3.5))


def _manhattan_center(sq: int) -> float:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return abs(f - 3.5) + abs(r - 3.5)


def _build_tables() -> tuple[list[list[int]], list[list[int]]]:
    """Return (mg, eg) tables indexed [piece_type][square] from White's perspective."""
    mg = [[0] * 64 for _ in range(7)]
    eg = [[0] * 64 for _ in range(7)]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        cd = _center_distance(sq)  # 0.5 .. 3.5
        md = _manhattan_center(sq)  # 1 .. 7
        # pawns: advance, centre files, penalise unmoved central pawns a little
        adv = r - 1  # 0 on rank 2
        centre = 1.0 if f in (3, 4) else (0.5 if f in (2, 5) else 0.0)
        p_mg = int(4 * adv + 10 * centre * (1 if 2 <= r <= 4 else 0) - (8 if (r == 1 and f in (3, 4)) else 0))
        p_eg = int(10 * adv + (20 if r >= 5 else 0) + (25 if r == 6 else 0))
        mg[chess.PAWN][sq] = p_mg
        eg[chess.PAWN][sq] = p_eg
        # knights: centralisation, rim penalty
        n_c = int(30 - 12 * cd)
        rim = -15 if (f in (0, 7) or r in (0, 7)) else 0
        mg[chess.KNIGHT][sq] = n_c + rim
        eg[chess.KNIGHT][sq] = int(20 - 8 * cd) + rim
        # bishops: long diagonals, mild centralisation
        b_c = int(15 - 5 * cd)
        diag = 8 if (f == r or f + r == 7) else 0
        mg[chess.BISHOP][sq] = b_c + diag - (10 if r == 0 and f in (2, 5) else 0)
        eg[chess.BISHOP][sq] = int(12 - 4 * cd) + diag
        # rooks: 7th rank, central files in mg
        mg[chess.ROOK][sq] = (12 if r == 6 else 0) + (6 if f in (3, 4) else 0) - (4 if f in (0, 7) else 0)
        eg[chess.ROOK][sq] = (8 if r == 6 else 0) + int(4 - md // 2)
        # queen: mild centralisation, discourage early development squares slightly
        mg[chess.QUEEN][sq] = int(8 - 3 * cd) - (5 if r >= 4 else 0)
        eg[chess.QUEEN][sq] = int(14 - 4 * cd)
        # king: shelter in mg (back rank corners), centre in eg
        k_mg = 0
        if r == 0:
            k_mg = 30 if f in (1, 2, 6, 7) else (10 if f in (0, 3, 4, 5) else 0)
        elif r == 1:
            k_mg = 5 if f in (0, 1, 6, 7) else -10
        else:
            k_mg = -20 * r
        mg[chess.KING][sq] = k_mg
        eg[chess.KING][sq] = int(30 - 10 * cd)
    return mg, eg


_MG, _EG = _build_tables()

# Full tables including material: [piece_type][color][square]; black uses the mirrored square.
MG_TABLE: list[list[list[int]]] = [[[0] * 64, [0] * 64] for _ in range(7)]
EG_TABLE: list[list[list[int]]] = [[[0] * 64, [0] * 64] for _ in range(7)]
for _pt in range(1, 7):
    for _sq in range(64):
        MG_TABLE[_pt][chess.WHITE][_sq] = PIECE_VALUE_MG[_pt] + _MG[_pt][_sq]
        EG_TABLE[_pt][chess.WHITE][_sq] = PIECE_VALUE_EG[_pt] + _EG[_pt][_sq]
        _m = chess.square_mirror(_sq)
        MG_TABLE[_pt][chess.BLACK][_sq] = -(PIECE_VALUE_MG[_pt] + _MG[_pt][_m])
        EG_TABLE[_pt][chess.BLACK][_sq] = -(PIECE_VALUE_EG[_pt] + _EG[_pt][_m])

_scan = chess.scan_forward


def evaluate(board: chess.Board) -> int:
    """Tapered evaluation from the side to move's point of view, in centipawns."""
    occ_w = board.occupied_co[chess.WHITE]
    occ_b = board.occupied_co[chess.BLACK]
    mg = 0
    eg = 0
    phase = 0
    for pt, bb in (
        (chess.PAWN, board.pawns),
        (chess.KNIGHT, board.knights),
        (chess.BISHOP, board.bishops),
        (chess.ROOK, board.rooks),
        (chess.QUEEN, board.queens),
        (chess.KING, board.kings),
    ):
        mgw = MG_TABLE[pt][chess.WHITE]
        egw = EG_TABLE[pt][chess.WHITE]
        mgb = MG_TABLE[pt][chess.BLACK]
        egb = EG_TABLE[pt][chess.BLACK]
        wb = bb & occ_w
        bbb = bb & occ_b
        if pt != chess.PAWN and pt != chess.KING:
            phase += PHASE_WEIGHT[pt] * ((wb.bit_count()) + (bbb.bit_count()))
        for sq in _scan(wb):
            mg += mgw[sq]
            eg += egw[sq]
        for sq in _scan(bbb):
            mg += mgb[sq]
            eg += egb[sq]
    # bishop pair
    if (board.bishops & occ_w).bit_count() >= 2:
        mg += 25
        eg += 40
    if (board.bishops & occ_b).bit_count() >= 2:
        mg -= 25
        eg -= 40
    if phase > MAX_PHASE:
        phase = MAX_PHASE
    score = int((mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE)
    # tempo
    score += 8 if board.turn == chess.WHITE else -8
    return score if board.turn == chess.WHITE else -score


def material_only(board: chess.Board) -> int:
    """Non-pawn material of the side to move, in mg centipawns (for null-move guards)."""
    occ = board.occupied_co[board.turn]
    return (
        PIECE_VALUE_MG[chess.KNIGHT] * (board.knights & occ).bit_count()
        + PIECE_VALUE_MG[chess.BISHOP] * (board.bishops & occ).bit_count()
        + PIECE_VALUE_MG[chess.ROOK] * (board.rooks & occ).bit_count()
        + PIECE_VALUE_MG[chess.QUEEN] * (board.queens & occ).bit_count()
    )
