"""Iterative deepening chess agent.

Implements my-agents-readmes/05_iterative_deepening.md on top of the earlier stages
(negamax, alpha-beta, a handcrafted evaluation and quiescence search):

    depth 1 -> depth 2 -> depth 3 -> ... until the time budget (or a depth cap) is hit

The root loop keeps the best move from the last *fully completed* iteration, so an
abort in the middle of depth d never hands back a half-searched answer. Each new
iteration searches the previous iteration's best move first, which is where most of
the alpha-beta cut-offs come from, and records the principal variation for debugging.

Two modes are exposed through `search()`:

    fixed depth   search(board, max_depth=4)                  no clock, always finishes
    timed         search(board, time_limit_s=1.0)             abort safely, keep last depth

`get_move` is the platform entry point and uses the timed mode.

Lives at my-agents/05_iterative_deepening/agent.py so the harness can import it. To
submit, copy it to agent.py at the root of the repo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess

# ---------------------------------------------------------------------------
# Evaluation (stage 03): material + piece-square tables, tapered by game phase
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
BISHOP_PAIR_BONUS = 30
DOUBLED_PAWN_PENALTY = 12
ISOLATED_PAWN_PENALTY = 10
# Passed pawn bonus indexed by the pawn's rank from its own side (0 = home rank).
PASSED_PAWN_BONUS = [0, 10, 15, 25, 45, 70, 110, 0]

# Tables are written as they appear on a diagram (rank 8 first, White at the bottom)
# and flipped below so index 0 is a1, matching python-chess square numbering.
# fmt: off
_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
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
_KING_MIDDLEGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
_KING_ENDGAME = [
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


def _mirror(table: list[int]) -> list[int]:
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_MG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING_MIDDLEGAME),
}
_WHITE_EG: dict[chess.PieceType, list[int]] = dict(_WHITE_MG)
_WHITE_EG[chess.KING] = _from_diagram(_KING_ENDGAME)

# PST_MG[colour][piece][square], likewise PST_EG. Black tables are the White tables mirrored.
PST_MG: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: _WHITE_MG,
    chess.BLACK: {piece: _mirror(table) for piece, table in _WHITE_MG.items()},
}
PST_EG: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: _WHITE_EG,
    chess.BLACK: {piece: _mirror(table) for piece, table in _WHITE_EG.items()},
}

# Game phase: weighted count of non-pawn material. 24 = opening, 0 = bare kings and pawns.
_PHASE_WEIGHT: dict[chess.PieceType, int] = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
_MAX_PHASE = 24

# Files adjacent to each file, for the isolated pawn test.
_ADJACENT_FILES: list[int] = [
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
]

# Squares a white pawn on `square` must have free of enemy pawns to be passed
# (same file and adjacent files, all ranks ahead). Black uses the mirrored table.
_WHITE_PASSED_MASK: list[int] = []
for _sq in chess.SQUARES:
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _ahead = 0
    for _rank in range(_r + 1, 8):
        _ahead |= chess.BB_RANKS[_rank]
    _WHITE_PASSED_MASK.append(_ahead & (chess.BB_FILES[_f] | _ADJACENT_FILES[_f]))
# A black pawn's mask is the white mask of the mirrored square, flipped back.
_BLACK_PASSED_MASK: list[int] = [
    chess.flip_vertical(_WHITE_PASSED_MASK[chess.square_mirror(sq)]) for sq in chess.SQUARES
]
_PASSED_MASK: dict[chess.Color, list[int]] = {
    chess.WHITE: _WHITE_PASSED_MASK,
    chess.BLACK: _BLACK_PASSED_MASK,
}


def _pawn_structure(board: chess.Board, colour: chess.Color) -> int:
    """Doubled and isolated penalties, passed pawn bonus, for one side, in centipawns."""
    own = board.pieces_mask(chess.PAWN, colour)
    enemy = board.pieces_mask(chess.PAWN, not colour)
    score = 0
    for file_index in range(8):
        on_file = chess.popcount(own & chess.BB_FILES[file_index])
        if on_file > 1:
            score -= DOUBLED_PAWN_PENALTY * (on_file - 1)
        if on_file and not own & _ADJACENT_FILES[file_index]:
            score -= ISOLATED_PAWN_PENALTY * on_file
    masks = _PASSED_MASK[colour]
    for square in chess.scan_forward(own):
        if not enemy & masks[square]:
            rank = chess.square_rank(square)
            relative_rank = rank if colour == chess.WHITE else 7 - rank
            score += PASSED_PAWN_BONUS[relative_rank]
    return score


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns from the perspective of the side to move.

    Computed as white - black, then flipped for Black, so mirrored positions score
    exactly opposite. Positive means the mover is better, which is the convention
    negamax needs so a child's score can simply be negated.
    """
    mg = 0
    eg = 0
    phase = 0
    for square, piece in board.piece_map().items():
        kind = piece.piece_type
        material = PIECE_VALUE[kind]
        sign = 1 if piece.color == chess.WHITE else -1
        mg += sign * (material + PST_MG[piece.color][kind][square])
        eg += sign * (material + PST_EG[piece.color][kind][square])
        phase += _PHASE_WEIGHT.get(kind, 0)

    phase = min(phase, _MAX_PHASE)
    score = (mg * phase + eg * (_MAX_PHASE - phase)) // _MAX_PHASE

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS

    score += _pawn_structure(board, chess.WHITE) - _pawn_structure(board, chess.BLACK)

    return score if board.turn == chess.WHITE else -score


# ---------------------------------------------------------------------------
# Move ordering: previous best move first, then MVV-LVA captures and promotions
# ---------------------------------------------------------------------------


def _move_priority(board: chess.Board, move: chess.Move) -> int:
    """Higher is searched earlier: promotions, then captures by MVV-LVA, then the rest."""
    priority = 0
    if move.promotion:
        priority += 10_000 + PIECE_VALUE[move.promotion]
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]  # en passant
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker else 0
        priority += 1_000 + 10 * victim_value - attacker_value
    return priority


def ordered_moves(board: chess.Board, first: chess.Move | None = None) -> list[chess.Move]:
    """All legal moves, best guess first. `first` (e.g. the previous PV move) leads."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: _move_priority(board, move), reverse=True)
    if first is not None and first in moves:
        moves.remove(first)
        moves.insert(0, first)
    return moves


# ---------------------------------------------------------------------------
# Search: negamax + alpha-beta + quiescence, with a deadline that can abort it
# ---------------------------------------------------------------------------

MATE_SCORE = 100_000
MAX_PLY = 128
MATE_THRESHOLD = MATE_SCORE - MAX_PLY  # any |score| above this is a proven mate
INFINITY = MATE_SCORE * 2
NODES_PER_CLOCK_CHECK = 256


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


def is_mate_score(score: int) -> bool:
    return abs(score) >= MATE_THRESHOLD


def format_score(score: int) -> str:
    """'+37' for centipawns, '#3' / '#-3' for mate in n moves."""
    if is_mate_score(score):
        plies = MATE_SCORE - abs(score)
        moves = (plies + 1) // 2
        return f"#{moves}" if score > 0 else f"#-{moves}"
    return f"{score:+d}"


class Searcher:
    """One search on one root position. Holds the deadline, node counter and PV."""

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.nodes = 0
        # Triangular PV: pv_table[ply] holds the best line found from that ply.
        self.pv_table: list[list[chess.Move]] = [[] for _ in range(MAX_PLY + 1)]

    def _tick(self) -> None:
        self.nodes += 1
        if (
            self.deadline is not None
            and self.nodes % NODES_PER_CLOCK_CHECK == 0
            and time.monotonic() > self.deadline
        ):
            raise OutOfTime

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Resolve captures and promotions so a leaf is never mid-exchange.

        In check the stand-pat score is meaningless, so every evasion is searched.
        """
        self._tick()
        if ply >= MAX_PLY:
            return evaluate(board)

        in_check = board.is_check()
        if not in_check:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best = stand_pat
        else:
            best = -INFINITY

        moves = ordered_moves(board)
        if in_check and not moves:
            return -(MATE_SCORE - ply)

        for move in moves:
            if not in_check and not board.is_capture(move) and not move.promotion:
                continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Best score the side to move can force within `depth` plies."""
        self._tick()
        self.pv_table[ply] = []

        # Draws are checked before move generation so we never "win" material in a
        # line the opponent can repeat or claim out of.
        if ply > 0 and (
            board.is_repetition(2)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            return 0

        if ply >= MAX_PLY:
            return evaluate(board)

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        moves = ordered_moves(board)
        if not moves:
            # Checkmated: closer mates score worse for the loser via `ply`. Stalemate: draw.
            return -(MATE_SCORE - ply) if board.is_check() else 0

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    self.pv_table[ply] = [move, *self.pv_table[ply + 1]]
                    if alpha >= beta:
                        break  # the opponent will never allow this line
        return best

    def search_root(
        self, board: chess.Board, depth: int, first: chess.Move | None = None
    ) -> tuple[chess.Move, int, list[chess.Move]]:
        """One full-width search at the root. Returns (best move, score, principal variation).

        `first` is searched before everything else; iterative deepening passes the
        previous iteration's best move so the new iteration starts with a good bound.
        """
        moves = ordered_moves(board, first)
        assert moves, "search_root called with no legal moves"

        best_move = moves[0]
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        pv: list[chess.Move] = [best_move]

        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
                pv = [move, *self.pv_table[1]]
                alpha = max(alpha, score)
        return best_move, best_score, pv


# ---------------------------------------------------------------------------
# Iterative deepening (stage 05) and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 40
MAX_BUDGET_MS = 8_000
SAFETY_MARGIN_MS = 150  # never plan to use the very last slice of the clock
# Do not start another iteration when this fraction of the budget is already spent:
# the next depth usually costs several times the previous one, so it would not finish.
NEXT_ITERATION_FRACTION = 0.45


@dataclass
class SearchResult:
    """What iterative deepening found, plus the statistics the spec asks for."""

    move: chess.Move
    score: int
    depth: int  # deepest fully completed iteration
    nodes: int
    time_s: float
    pv: list[chess.Move] = field(default_factory=list)
    completed_depths: list[int] = field(default_factory=list)
    aborted: bool = False  # the last (unfinished) iteration was cut off by the clock

    def pv_uci(self) -> str:
        return " ".join(move.uci() for move in self.pv)


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    budget = max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))
    return max(1, min(budget, time_left_ms - SAFETY_MARGIN_MS))


def search(
    board: chess.Board,
    max_depth: int = MAX_DEPTH,
    time_limit_s: float | None = None,
    verbose: bool = False,
) -> SearchResult:
    """Iterative deepening: search depth 1, 2, 3 ... and keep the last finished depth.

    fixed-depth mode   time_limit_s is None: every depth up to max_depth completes.
    timed mode         the iteration running when the clock expires is abandoned and
                       the result of the previous, complete iteration is returned.

    Depth 1 is never aborted so there is always a legal answer. Each iteration
    searches the previous best move first. Deepening stops early once a forced
    mate is proven, since more depth cannot change that answer.
    """
    started = time.monotonic()
    deadline = None if time_limit_s is None else started + time_limit_s
    searcher = Searcher(deadline=None)  # depth 1 must always finish
    root_ply = len(board.move_stack)  # an abort unwinds through pushed moves; pop back to here

    best_move, best_score, pv = searcher.search_root(board, 1)
    result = SearchResult(
        move=best_move,
        score=best_score,
        depth=1,
        nodes=searcher.nodes,
        time_s=time.monotonic() - started,
        pv=pv,
        completed_depths=[1],
    )
    _report(result, verbose)

    searcher.deadline = deadline
    for depth in range(2, max_depth + 1):
        if is_mate_score(result.score):
            break  # forced mate proven; deeper search cannot improve on it
        if deadline is not None and time_limit_s is not None:
            elapsed = time.monotonic() - started
            if elapsed > NEXT_ITERATION_FRACTION * time_limit_s:
                break
        try:
            best_move, best_score, pv = searcher.search_root(board, depth, first=result.move)
        except OutOfTime:
            while len(board.move_stack) > root_ply:
                board.pop()
            result.aborted = True
            break
        result.move = best_move
        result.score = best_score
        result.depth = depth
        result.pv = pv
        result.completed_depths.append(depth)
        result.nodes = searcher.nodes
        result.time_s = time.monotonic() - started
        _report(result, verbose)

    result.nodes = searcher.nodes
    result.time_s = time.monotonic() - started
    return result


def _report(result: SearchResult, verbose: bool) -> None:
    if verbose:
        print(
            f"depth={result.depth} score={format_score(result.score)} "
            f"nodes={result.nodes} time={result.time_s:.2f}s pv={result.pv_uci()}"
        )


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    budget_s = move_budget_ms(time_left_ms) / 1000.0
    result = search(board, time_limit_s=budget_s, verbose=True)
    return result.move.uci()
