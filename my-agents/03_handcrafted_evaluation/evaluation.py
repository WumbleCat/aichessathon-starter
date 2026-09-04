"""Handcrafted, deterministic position evaluation in centipawns.

Follows my-agents-readmes/03_handcrafted_evaluation.md:

    evaluation = material
               + piece-square tables   (separate middlegame / endgame king tables)
               + bishop pair
               + pawn structure        (doubled, isolated, backward, passed, connected passed)
               + mobility              (pseudo-legal attacks, minus own pieces and squares
                                        guarded by enemy pawns)
               + king safety           (pawn shield, open files near the king, attackers
                                        near the king)
               + rook activity         (open / semi-open file, seventh rank, connected rooks)

Every term is computed for White and for Black, the totals are blended between a
middlegame and an endgame score by the amount of non-pawn material left, and the
result is `white - black`. `evaluate` flips that to the side-to-move perspective
Negamax needs.

No learned weights. Everything is a table or a small constant, and all of them are
plain module-level names so they can be tuned.

Performance notes: the hot loop works on python-chess bitboards (plain ints) and
precomputed masks. It never copies the board, never builds a FEN, and the pawn
structure term is cached by pawn configuration because it only changes on pawn moves.
"""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

PAWN_VALUE = 100
KNIGHT_VALUE = 320
BISHOP_VALUE = 330
ROOK_VALUE = 500
QUEEN_VALUE = 900

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: PAWN_VALUE,
    chess.KNIGHT: KNIGHT_VALUE,
    chess.BISHOP: BISHOP_VALUE,
    chess.ROOK: ROOK_VALUE,
    chess.QUEEN: QUEEN_VALUE,
    chess.KING: 0,  # the king is never traded, so it carries no material value
}

# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------

# Weighted count of non-pawn material. A full starting army adds up to TOTAL_PHASE;
# with nothing but pawns and kings left the phase is 0 and the endgame tables rule.
PHASE_WEIGHT: dict[chess.PieceType, int] = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
TOTAL_PHASE = 24  # 4 minors, 4 rooks, 2 queens

# ---------------------------------------------------------------------------
# Piece-square tables
# ---------------------------------------------------------------------------
# Written as they appear on a diagram with White at the bottom (rank 8 first, a-file
# left). `_from_diagram` reorders them so index 0 is a1, as python-chess counts squares.
# Black uses the same tables through a vertical mirror instead of duplicated values.

# fmt: off
_PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    15, 15, 15, 15, 15, 15, 15, 15,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
# Middlegame king: stay castled behind pawns, avoid the centre.
_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
# Endgame king: walk to the centre and help the pawns.
_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]
# fmt: on


def _from_diagram(table: list[int]) -> list[int]:
    """Reorder a diagram-style table (rank 8 first) into a1..h8 square order."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


def _mirrored(table: list[int]) -> list[int]:
    """The same a1..h8 table seen from Black's side of the board."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_MG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN_MG),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING_MG),
}
_WHITE_EG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN_EG),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING_EG),
}

# PST_MG[colour][piece_type][square], likewise PST_EG. Material is folded in so the hot
# loop does one table lookup per piece instead of two.
PST_MG: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: {p: [PIECE_VALUE[p] + v for v in t] for p, t in _WHITE_MG.items()},
    chess.BLACK: {p: [PIECE_VALUE[p] + v for v in _mirrored(t)] for p, t in _WHITE_MG.items()},
}
PST_EG: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: {p: [PIECE_VALUE[p] + v for v in t] for p, t in _WHITE_EG.items()},
    chess.BLACK: {p: [PIECE_VALUE[p] + v for v in _mirrored(t)] for p, t in _WHITE_EG.items()},
}

# ---------------------------------------------------------------------------
# Tunable bonuses and penalties, as (middlegame, endgame) pairs
# ---------------------------------------------------------------------------

BISHOP_PAIR = (30, 50)

DOUBLED_PAWN = (-10, -20)
ISOLATED_PAWN = (-15, -15)
BACKWARD_PAWN = (-8, -12)
# Indexed by the pawn's relative rank (0 = own back rank, never used; 7 = promotion).
PASSED_PAWN_MG = [0, 5, 10, 20, 35, 60, 100, 0]
PASSED_PAWN_EG = [0, 10, 20, 35, 60, 100, 150, 0]
CONNECTED_PASSED = (10, 20)

# Per attacked square that is neither occupied by a friendly piece nor guarded by an
# enemy pawn.
MOBILITY_WEIGHT: dict[chess.PieceType, tuple[int, int]] = {
    chess.KNIGHT: (4, 4),
    chess.BISHOP: (5, 5),
    chess.ROOK: (2, 4),
    chess.QUEEN: (1, 2),
}

ROOK_OPEN_FILE = (25, 15)
ROOK_SEMI_OPEN_FILE = (12, 8)
ROOK_ON_SEVENTH = (20, 30)
CONNECTED_ROOKS = (10, 5)

# King safety is a middlegame concern; it fades out with the phase.
SHIELD_PAWN_NEAR = 12  # friendly pawn one rank ahead of the king, same or adjacent file
SHIELD_PAWN_FAR = 6  # friendly pawn two ranks ahead
OPEN_FILE_NEAR_KING = -20  # no pawns at all on the king's file or a neighbour
SEMI_OPEN_FILE_NEAR_KING = -10  # no friendly pawn on the king's file or a neighbour
# Enemy pieces attacking the king zone accumulate "attack units"; the table turns units
# into a penalty that grows faster than linearly, so several attackers hurt a lot.
ATTACK_UNITS: dict[chess.PieceType, int] = {
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
}
KING_ATTACK_PENALTY = [0, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135]
KING_ATTACK_PENALTY += [150] * 32  # room for every piece attacking at once

# ---------------------------------------------------------------------------
# Precomputed masks
# ---------------------------------------------------------------------------

_ALL = chess.BB_ALL
FILE_MASK = list(chess.BB_FILES)
ADJACENT_FILES = [
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
]


def _front_span(color: chess.Color, square: chess.Square) -> int:
    """Squares on this file strictly ahead of `square` from `color`'s point of view."""
    rank = chess.square_rank(square)
    file_mask = chess.BB_FILES[chess.square_file(square)]
    ahead = (_ALL << (8 * (rank + 1))) & _ALL if color == chess.WHITE else _ALL >> (8 * (8 - rank))
    return file_mask & ahead


def _widen(mask: int) -> int:
    """Shift a mask one file to each side (the squares stay on the board)."""
    return ((mask << 1) & ~chess.BB_FILE_A) | ((mask >> 1) & ~chess.BB_FILE_H)


# FRONT_SPAN[colour][square]: the file ahead of a pawn (doubled-pawn test).
FRONT_SPAN: list[list[int]] = [[0] * 64, [0] * 64]
# PASSED_MASK[colour][square]: a pawn is passed when no enemy pawn is in this mask.
PASSED_MASK: list[list[int]] = [[0] * 64, [0] * 64]
# SUPPORT_MASK[colour][square]: squares on adjacent files level with or behind the pawn,
# where a friendly pawn could still come to support it. If none is there and the stop
# square is guarded by an enemy pawn, the pawn is backward.
SUPPORT_MASK: list[list[int]] = [[0] * 64, [0] * 64]
# STOP_SQUARE[colour][square]: the square directly in front of the pawn.
STOP_SQUARE: list[list[int]] = [[0] * 64, [0] * 64]
# SHIELD_NEAR / SHIELD_FAR [colour][king square]: the three squares one / two ranks ahead.
SHIELD_NEAR: list[list[int]] = [[0] * 64, [0] * 64]
SHIELD_FAR: list[list[int]] = [[0] * 64, [0] * 64]
# KING_ZONE[square]: the king and its neighbours.
KING_ZONE: list[int] = [chess.BB_KING_ATTACKS[sq] | chess.BB_SQUARES[sq] for sq in chess.SQUARES]


def _build_masks() -> None:
    for color in (chess.WHITE, chess.BLACK):
        forward = 1 if color == chess.WHITE else -1
        for sq in chess.SQUARES:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            front = _front_span(color, sq)
            FRONT_SPAN[color][sq] = front
            PASSED_MASK[color][sq] = (front | _widen(front)) & _ALL
            rear = _front_span(not color, sq) | chess.BB_SQUARES[sq]
            SUPPORT_MASK[color][sq] = _widen(rear) & ADJACENT_FILES[file] & _ALL
            stop = sq + 8 * forward
            STOP_SQUARE[color][sq] = chess.BB_SQUARES[stop] if 0 <= stop < 64 else 0
            three_files = FILE_MASK[file] | ADJACENT_FILES[file]
            near_rank = rank + forward
            far_rank = rank + 2 * forward
            if 0 <= near_rank < 8:
                SHIELD_NEAR[color][sq] = chess.BB_RANKS[near_rank] & three_files
            if 0 <= far_rank < 8:
                SHIELD_FAR[color][sq] = chess.BB_RANKS[far_rank] & three_files


_build_masks()

# RELATIVE_RANK[colour][square]: 0 on the colour's own back rank, 7 on the far rank.
RELATIVE_RANK: list[list[int]] = [
    [7 - chess.square_rank(sq) for sq in chess.SQUARES],  # index 0 = BLACK
    [chess.square_rank(sq) for sq in chess.SQUARES],  # index 1 = WHITE
]
SEVENTH_RANK: list[int] = [chess.BB_RANK_2, chess.BB_RANK_7]  # relative seventh, per colour

_scan = chess.scan_forward
_popcount = chess.popcount


def pawn_attacks(pawns: int, color: chess.Color) -> int:
    """Every square attacked by these pawns of `color`."""
    if color == chess.WHITE:
        return (((pawns << 7) & ~chess.BB_FILE_H) | ((pawns << 9) & ~chess.BB_FILE_A)) & _ALL
    return ((pawns >> 7) & ~chess.BB_FILE_A) | ((pawns >> 9) & ~chess.BB_FILE_H)


# ---------------------------------------------------------------------------
# Pawn structure (cached by pawn configuration)
# ---------------------------------------------------------------------------

PAWN_CACHE_LIMIT = 200_000
_pawn_cache: dict[tuple[int, int], tuple[int, int]] = {}


def _pawn_structure_one_side(own: int, enemy: int, color: chess.Color) -> tuple[int, int]:
    """(middlegame, endgame) pawn-structure score for one colour; positive is good."""
    mg = eg = 0
    enemy_guarded = pawn_attacks(enemy, not color)
    front_span = FRONT_SPAN[color]
    passed_mask = PASSED_MASK[color]
    support_mask = SUPPORT_MASK[color]
    stop_square = STOP_SQUARE[color]
    relative_rank = RELATIVE_RANK[color]
    passed_squares = 0

    for square in _scan(own):
        file = chess.square_file(square)
        # Doubled: another friendly pawn on the same file ahead of this one. Counting
        # only pawns ahead charges a stack of n pawns n-1 times.
        if own & front_span[square]:
            mg += DOUBLED_PAWN[0]
            eg += DOUBLED_PAWN[1]
        adjacent = own & ADJACENT_FILES[file]
        if not adjacent:
            mg += ISOLATED_PAWN[0]
            eg += ISOLATED_PAWN[1]
        if not enemy & passed_mask[square]:
            rank = relative_rank[square]
            mg += PASSED_PAWN_MG[rank]
            eg += PASSED_PAWN_EG[rank]
            passed_squares |= chess.BB_SQUARES[square]
        elif adjacent and not own & support_mask[square] and enemy_guarded & stop_square[square]:
            # Backward: has neighbours but they are all ahead of it, and it cannot
            # advance because an enemy pawn controls the square in front.
            mg += BACKWARD_PAWN[0]
            eg += BACKWARD_PAWN[1]

    # Connected passed pawns: a passed pawn with a friendly pawn beside or diagonally
    # next to it (the squares a king on that pawn's square would attack, adjacent files).
    for square in _scan(passed_squares):
        neighbours = chess.BB_KING_ATTACKS[square] & ADJACENT_FILES[chess.square_file(square)]
        if own & neighbours:
            mg += CONNECTED_PASSED[0]
            eg += CONNECTED_PASSED[1]
    return mg, eg


def pawn_structure(white_pawns: int, black_pawns: int) -> tuple[int, int]:
    """(middlegame, endgame) pawn-structure score, White minus Black. Cached."""
    key = (white_pawns, black_pawns)
    cached = _pawn_cache.get(key)
    if cached is not None:
        return cached
    w_mg, w_eg = _pawn_structure_one_side(white_pawns, black_pawns, chess.WHITE)
    b_mg, b_eg = _pawn_structure_one_side(black_pawns, white_pawns, chess.BLACK)
    result = (w_mg - b_mg, w_eg - b_eg)
    if len(_pawn_cache) >= PAWN_CACHE_LIMIT:
        _pawn_cache.clear()
    _pawn_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Pieces: material, PST, mobility, bishop pair, rook activity, king attackers
# ---------------------------------------------------------------------------


def _pieces_one_side(
    board: chess.Board, color: chess.Color, own_pawns: int, enemy_pawns: int
) -> tuple[int, int, int, int]:
    """Score every non-pawn, non-king piece of `color`.

    Returns (mg, eg, phase contribution, attack units against the enemy king).
    """
    mg = eg = phase = attack_units = 0
    pst_mg = PST_MG[color]
    pst_eg = PST_EG[color]
    own = board.occupied_co[color]
    occupied = board.occupied
    # Mobility counts squares that are not ours and not guarded by an enemy pawn.
    mobility_area = ~(own | pawn_attacks(enemy_pawns, not color)) & _ALL
    enemy_king = board.king(not color)
    enemy_zone = KING_ZONE[enemy_king] if enemy_king is not None else 0
    all_pawns = own_pawns | enemy_pawns

    knights = board.knights & own
    bishops = board.bishops & own
    rooks = board.rooks & own
    queens = board.queens & own

    knight_mg, knight_eg = MOBILITY_WEIGHT[chess.KNIGHT]
    for square in _scan(knights):
        mg += pst_mg[chess.KNIGHT][square]
        eg += pst_eg[chess.KNIGHT][square]
        attacks = chess.BB_KNIGHT_ATTACKS[square]
        moves = _popcount(attacks & mobility_area)
        mg += knight_mg * moves
        eg += knight_eg * moves
        if attacks & enemy_zone:
            attack_units += ATTACK_UNITS[chess.KNIGHT]
    phase += PHASE_WEIGHT[chess.KNIGHT] * _popcount(knights)

    bishop_mg, bishop_eg = MOBILITY_WEIGHT[chess.BISHOP]
    for square in _scan(bishops):
        mg += pst_mg[chess.BISHOP][square]
        eg += pst_eg[chess.BISHOP][square]
        attacks = chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
        moves = _popcount(attacks & mobility_area)
        mg += bishop_mg * moves
        eg += bishop_eg * moves
        if attacks & enemy_zone:
            attack_units += ATTACK_UNITS[chess.BISHOP]
    bishop_count = _popcount(bishops)
    phase += PHASE_WEIGHT[chess.BISHOP] * bishop_count
    if bishop_count >= 2:
        mg += BISHOP_PAIR[0]
        eg += BISHOP_PAIR[1]

    rook_mg, rook_eg = MOBILITY_WEIGHT[chess.ROOK]
    seventh = SEVENTH_RANK[color]
    for square in _scan(rooks):
        mg += pst_mg[chess.ROOK][square]
        eg += pst_eg[chess.ROOK][square]
        attacks = (
            chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
            | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]
        )
        moves = _popcount(attacks & mobility_area)
        mg += rook_mg * moves
        eg += rook_eg * moves
        if attacks & enemy_zone:
            attack_units += ATTACK_UNITS[chess.ROOK]
        file_mask = FILE_MASK[chess.square_file(square)]
        if not all_pawns & file_mask:
            mg += ROOK_OPEN_FILE[0]
            eg += ROOK_OPEN_FILE[1]
        elif not own_pawns & file_mask:
            mg += ROOK_SEMI_OPEN_FILE[0]
            eg += ROOK_SEMI_OPEN_FILE[1]
        if chess.BB_SQUARES[square] & seventh:
            mg += ROOK_ON_SEVENTH[0]
            eg += ROOK_ON_SEVENTH[1]
        # Connected rooks: this rook's line attacks reach another friendly rook. Each
        # pair is seen from both rooks, so award half from each side.
        if attacks & rooks:
            mg += CONNECTED_ROOKS[0] // 2
            eg += CONNECTED_ROOKS[1] // 2
    phase += PHASE_WEIGHT[chess.ROOK] * _popcount(rooks)

    queen_mg, queen_eg = MOBILITY_WEIGHT[chess.QUEEN]
    for square in _scan(queens):
        mg += pst_mg[chess.QUEEN][square]
        eg += pst_eg[chess.QUEEN][square]
        attacks = (
            chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
            | chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
            | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]
        )
        moves = _popcount(attacks & mobility_area)
        mg += queen_mg * moves
        eg += queen_eg * moves
        if attacks & enemy_zone:
            attack_units += ATTACK_UNITS[chess.QUEEN]
    phase += PHASE_WEIGHT[chess.QUEEN] * _popcount(queens)

    return mg, eg, phase, attack_units


# ---------------------------------------------------------------------------
# King safety (middlegame only; the phase blend scales it away in the endgame)
# ---------------------------------------------------------------------------


def _king_shelter(
    king: chess.Square | None, color: chess.Color, own_pawns: int, enemy_pawns: int
) -> int:
    """Pawn shield and open files around `color`'s king, in middlegame centipawns."""
    if king is None:
        return 0
    score = SHIELD_PAWN_NEAR * _popcount(own_pawns & SHIELD_NEAR[color][king])
    score += SHIELD_PAWN_FAR * _popcount(own_pawns & SHIELD_FAR[color][king])
    file = chess.square_file(king)
    for f in range(max(0, file - 1), min(7, file + 1) + 1):
        file_mask = FILE_MASK[f]
        if not own_pawns & file_mask:
            if not enemy_pawns & file_mask:
                score += OPEN_FILE_NEAR_KING
            else:
                score += SEMI_OPEN_FILE_NEAR_KING
    return score


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def game_phase(board: chess.Board) -> int:
    """Non-pawn material left, 0 (bare kings and pawns) .. TOTAL_PHASE (full armies)."""
    phase = 0
    for piece_type, weight in PHASE_WEIGHT.items():
        phase += weight * _popcount(board.pieces_mask(piece_type, chess.WHITE))
        phase += weight * _popcount(board.pieces_mask(piece_type, chess.BLACK))
    return min(TOTAL_PHASE, phase)


def evaluate_white(board: chess.Board) -> int:
    """Static evaluation in centipawns from White's point of view. Positive favours White."""
    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]

    mg = eg = 0

    # Pawns: material and PST straight from the tables, then the cached structure term.
    table_mg = PST_MG[chess.WHITE][chess.PAWN]
    table_eg = PST_EG[chess.WHITE][chess.PAWN]
    for square in _scan(white_pawns):
        mg += table_mg[square]
        eg += table_eg[square]
    table_mg = PST_MG[chess.BLACK][chess.PAWN]
    table_eg = PST_EG[chess.BLACK][chess.PAWN]
    for square in _scan(black_pawns):
        mg -= table_mg[square]
        eg -= table_eg[square]
    structure_mg, structure_eg = pawn_structure(white_pawns, black_pawns)
    mg += structure_mg
    eg += structure_eg

    # Kings: placement only here; safety comes after the pieces are counted.
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is not None:
        mg += PST_MG[chess.WHITE][chess.KING][white_king]
        eg += PST_EG[chess.WHITE][chess.KING][white_king]
    if black_king is not None:
        mg -= PST_MG[chess.BLACK][chess.KING][black_king]
        eg -= PST_EG[chess.BLACK][chess.KING][black_king]

    # Pieces.
    w_mg, w_eg, w_phase, w_attack = _pieces_one_side(board, chess.WHITE, white_pawns, black_pawns)
    b_mg, b_eg, b_phase, b_attack = _pieces_one_side(board, chess.BLACK, black_pawns, white_pawns)
    mg += w_mg - b_mg
    eg += w_eg - b_eg

    # King safety: each king's shelter, and the pressure the enemy pieces put on it.
    mg += _king_shelter(white_king, chess.WHITE, white_pawns, black_pawns)
    mg -= _king_shelter(black_king, chess.BLACK, black_pawns, white_pawns)
    mg += KING_ATTACK_PENALTY[w_attack]  # White's pieces attacking Black's king
    mg -= KING_ATTACK_PENALTY[b_attack]  # Black's pieces attacking White's king

    # Blend by game phase. Truncate toward zero rather than floor so a mirrored
    # position scores exactly the negative.
    phase = min(TOTAL_PHASE, w_phase + b_phase)
    blended = mg * phase + eg * (TOTAL_PHASE - phase)
    return blended // TOTAL_PHASE if blended >= 0 else -((-blended) // TOTAL_PHASE)


def evaluate(board: chess.Board) -> int:
    """Static evaluation from the side to move's point of view, as Negamax expects."""
    score = evaluate_white(board)
    return score if board.turn == chess.WHITE else -score
