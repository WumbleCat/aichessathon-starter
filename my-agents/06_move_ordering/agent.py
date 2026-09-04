"""Move-ordering chess agent (stage 06 of my-agents-readmes).

The search is plain negamax with alpha-beta, quiescence and iterative deepening. The
point of this stage is `move_order_score`: rank the legal moves so that the moves most
likely to be best are searched first, which makes alpha-beta cut off earlier. Ordering
never changes the value alpha-beta returns at the root; it only changes how much work
it takes to get there.

Priority tiers, highest first (see my-agents-readmes/06_move_ordering.md):

    1. transposition-table / hint move   (supplied by the caller, e.g. last iteration's best)
    2. winning captures, MVV-LVA         (most valuable victim, least valuable attacker)
    3. promotions                        (queen promotion very high, underpromotions lower)
    4. killer moves                      (hook; quiet moves that recently caused a cutoff)
    5. quiet moves by history score      (hook; falls back to a check bonus + PST delta)
    6. losing captures                   (attacker worth more than the victim, square defended)

Killer and history tables are accepted as optional inputs so later stages can plug them
in without touching the scorer. Without SEE, "losing" captures are approximated cheaply.

The searcher records `nodes`, `beta_cutoffs` and `first_move_cutoffs` so the effect of
ordering is measurable: good ordering raises first_move_cutoffs / beta_cutoffs.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------------------
# Evaluation: material + piece-square tables, from the mover's point of view
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Diagram order (rank 8 first, White at the bottom); flipped below to a1..h8 indexing.
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
_KING = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
# fmt: on


def _from_diagram(table: list[int]) -> list[int]:
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_PST: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING),
}
_BLACK_PST: dict[chess.PieceType, list[int]] = {
    piece: [table[chess.square_mirror(square)] for square in chess.SQUARES]
    for piece, table in _WHITE_PST.items()
}
PST: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: _WHITE_PST,
    chess.BLACK: _BLACK_PST,
}

MATE_SCORE = 100_000
INFINITY = MATE_SCORE * 10
MAX_QUIESCENCE_PLY = 8  # bound the capture search; a real exchange is over well before this


def evaluate(board: chess.Board) -> int:
    """Static score for the side to move; positive means the mover is better."""
    score = 0
    turn = board.turn
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == turn else -value
    return score


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

# Tier bases. Tiers are far enough apart that no within-tier bonus can cross into the
# tier above. Within a tier, bigger is searched first.
SCORE_TT_MOVE = 10_000_000
SCORE_WINNING_CAPTURE = 8_000_000
SCORE_PROMOTION = 6_000_000
SCORE_KILLER = 4_000_000
SCORE_QUIET = 2_000_000
SCORE_LOSING_CAPTURE = 0

CHECK_BONUS = 500  # moderate: below any capture or promotion, above ordinary quiet moves
KILLER_SLOTS = 2

# History scores are keyed by (colour, piece type, to-square). Empty by default; a later
# stage fills it in and the scorer picks it up unchanged.
HistoryKey = tuple[chess.Color, chess.PieceType, chess.Square]
HistoryTable = Mapping[HistoryKey, int]
Killers = Sequence[chess.Move]


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square)
    # En passant lands on an empty square; the victim is always a pawn.
    return PIECE_VALUE[victim] if victim is not None else PIECE_VALUE[chess.PAWN]


def _attacker_value(board: chess.Board, move: chess.Move) -> int:
    attacker = board.piece_type_at(move.from_square)
    return PIECE_VALUE[attacker] if attacker is not None else 0


def mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Most Valuable Victim / Least Valuable Attacker score for a capture."""
    # 10x on the victim so a queen taken by a queen still beats a pawn taken by a pawn.
    return 10 * _victim_value(board, move) - _attacker_value(board, move)


def capture_looks_losing(board: chess.Board, move: chess.Move) -> bool:
    """Cheap stand-in for SEE: the attacker is worth more than the victim and the target
    square is defended. A real static exchange evaluation replaces this later."""
    if board.piece_type_at(move.from_square) == chess.KING:
        return False  # a legal king capture lands on an undefended square
    if _attacker_value(board, move) <= _victim_value(board, move):
        return False
    return board.is_attacked_by(not board.turn, move.to_square)


def move_order_score(
    board: chess.Board,
    move: chess.Move,
    tt_move: chess.Move | None = None,
    killers: Killers = (),
    history: HistoryTable | None = None,
) -> int:
    """Higher scores are searched first. Pure function of the position and the hints."""
    if move == tt_move:
        return SCORE_TT_MOVE

    if board.is_capture(move):
        score = mvv_lva(board, move)
        if move.promotion:
            # A capturing promotion is both; rank it with captures plus the piece gained.
            score += PIECE_VALUE[move.promotion]
        if capture_looks_losing(board, move):
            return SCORE_LOSING_CAPTURE + score
        return SCORE_WINNING_CAPTURE + score

    if move.promotion:
        # Queen promotions top the tier; underpromotions stay in the tier so they are
        # still tried before quiet moves, just after the queen.
        return SCORE_PROMOTION + PIECE_VALUE[move.promotion]

    if move in killers:
        return SCORE_KILLER + (KILLER_SLOTS - killers.index(move))  # earlier slot first

    score = SCORE_QUIET
    piece = board.piece_type_at(move.from_square)
    if piece is not None:
        if history:
            score += history.get((board.turn, piece, move.to_square), 0)
        # Small positional tie-breaker: prefer moves that improve the piece's square.
        table = PST[board.turn][piece]
        score += table[move.to_square] - table[move.from_square]
    if board.gives_check(move):
        score += CHECK_BONUS
    return score


def ordered_moves(
    board: chess.Board,
    tt_move: chess.Move | None = None,
    killers: Killers = (),
    history: HistoryTable | None = None,
) -> list[chess.Move]:
    """Every legal move exactly once, best guess first."""
    moves = list(board.legal_moves)
    moves.sort(
        key=lambda move: move_order_score(board, move, tt_move, killers, history),
        reverse=True,
    )
    return moves


# ---------------------------------------------------------------------------
# Search: negamax + alpha-beta + quiescence, with cutoff instrumentation
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


@dataclass
class SearchStats:
    nodes: int = 0
    qnodes: int = 0
    beta_cutoffs: int = 0
    first_move_cutoffs: int = 0

    @property
    def first_move_cutoff_rate(self) -> float:
        return self.first_move_cutoffs / self.beta_cutoffs if self.beta_cutoffs else 0.0


class Searcher:
    """One search for one root position.

    `order` toggles move ordering so that a fixed-depth search with and without it can
    be compared: the root score must match, the node count should not.
    """

    def __init__(self, deadline: float | None = None, order: bool = True) -> None:
        self.deadline = deadline
        self.order = order
        self.stats = SearchStats()
        self.killers: list[list[chess.Move]] = []  # per ply, newest first
        self.history: dict[HistoryKey, int] = {}  # unused here; wired for a later stage

    # -- helpers ------------------------------------------------------------

    def _tick(self) -> None:
        self.stats.nodes += 1
        if (
            self.deadline is not None
            and self.stats.nodes & 1023 == 0
            and time.monotonic() > self.deadline
        ):
            raise OutOfTime

    def _moves(self, board: chess.Board, ply: int, tt_move: chess.Move | None) -> list[chess.Move]:
        if not self.order:
            return list(board.legal_moves)
        killers: Killers = self.killers[ply] if ply < len(self.killers) else ()
        return ordered_moves(board, tt_move, killers, self.history)

    def _record_cutoff(self, board: chess.Board, move: chess.Move, ply: int, index: int) -> None:
        self.stats.beta_cutoffs += 1
        if index == 0:
            self.stats.first_move_cutoffs += 1
        if board.is_capture(move) or move.promotion:
            return
        # Remember the quiet move that refuted this line, for sibling nodes at this ply.
        while len(self.killers) <= ply:
            self.killers.append([])
        slot = self.killers[ply]
        if move not in slot:
            slot.insert(0, move)
            del slot[KILLER_SLOTS:]

    # -- quiescence -----------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, qdepth: int = 0) -> int:
        self._tick()
        self.stats.qnodes += 1
        stand_pat = evaluate(board)
        if stand_pat >= beta or qdepth >= MAX_QUIESCENCE_PLY:
            return stand_pat
        alpha = max(alpha, stand_pat)

        best = stand_pat
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if self.order:
            moves.sort(key=lambda m: move_order_score(board, m), reverse=True)
        for move in moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, qdepth + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- main search ----------------------------------------------------------

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()

        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        moves = self._moves(board, ply, None)
        if not moves:
            return -(MATE_SCORE + depth) if board.is_check() else 0

        best = -INFINITY
        for index, move in enumerate(moves):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        self._record_cutoff(board, move, ply, index)
                        break
        return best

    def search_root(
        self, board: chess.Board, depth: int, hint: chess.Move | None = None
    ) -> tuple[chess.Move, int]:
        """Full-window search at the root; `hint` (last iteration's best) goes first."""
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY

        for move in self._moves(board, 0, hint):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        if best_move is None:
            raise ValueError("search_root called with no legal moves")
        return best_move, best_score


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30
MIN_BUDGET_MS = 40
MAX_BUDGET_MS = 8_000
SAFETY_MS = 150  # never plan to use the last slice of the clock


def move_budget_ms(time_left_ms: int) -> int:
    budget = time_left_ms // MOVES_TO_GO
    budget = min(budget, max(0, time_left_ms - SAFETY_MS))
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(board: chess.Board, time_left_ms: int) -> chess.Move:
    """Deepen until the budget runs out; return the last fully completed depth's move."""
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(deadline=None)

    # Depth 1 runs with no deadline so there is always a completed answer to return,
    # even when the budget is tiny and quiescence runs long.
    best_move, best_score = searcher.search_root(board, 1)
    searcher.deadline = deadline
    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth, hint=best_move)
        except OutOfTime:
            break
        stats = searcher.stats
        print(
            f"depth {depth:2d} score {best_score:7d} best {best_move.uci()} "
            f"nodes {stats.nodes} cutoffs {stats.beta_cutoffs} "
            f"first {stats.first_move_cutoff_rate:.0%}"
        )
        if abs(best_score) >= MATE_SCORE:
            break
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
