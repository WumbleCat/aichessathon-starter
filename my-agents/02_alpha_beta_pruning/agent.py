"""Alpha-beta pruning chess agent.

Implements my-agents-readmes/02_alpha_beta_pruning.md on top of the plain Negamax of
01_negamax.md:

- ``negamax_plain``  the reference search from 01, kept so the two can be compared
- ``negamax``        the same search with an (alpha, beta) window; a node stops looking at
                     its remaining moves as soon as ``alpha >= beta``
- ``search_root``    full window at the root, tracks the best move separately
- ``SearchStats``    ``nodes_searched`` and ``beta_cutoffs``, reset before every root search

Both searches visit moves in ``board.legal_moves`` order and break ties by "first strictly
better", so at the same depth they return the same score and the same move. Alpha-beta only
visits fewer nodes. Move ordering, quiescence and iterative deepening are later steps and are
deliberately left out here.

Lives at my-agents/02_alpha_beta_pruning/agent.py so the harness can import it. To submit,
copy it to agent.py at the root of the repo.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------------------
# Evaluation: material plus piece-square tables, from the side to move's view
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Tables are written as they look on a diagram with White at the bottom (rank 8 first)
# and flipped below so that index 0 is a1, matching python-chess square numbering.
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


def _from_diagram(table: list[int]) -> list[int]:
    """Reorder a diagram-style table (rank 8 first) into a1..h8 square order."""
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

MATE_SCORE = 100_000  # larger than any material swing
INFINITY = math.inf


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move. Positive = mover is better."""
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == board.turn else -value
    return score


PositionKey = tuple[str, chess.Color, chess.Bitboard, chess.Square | None]

# Positions this game has already been through. get_move only receives a FEN, so the
# history is rebuilt from the FENs we are sent and the moves we answer with. The process
# lives for exactly one game, so module state is per game. Any position in here is
# scored as a draw by the search: a side that is ahead steers away from repeating, a
# side that is behind is happy to.
_game_history: set[PositionKey] = set()
_last_fullmove = 0


def position_key(board: chess.Board) -> PositionKey:
    return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)


def remember(board: chess.Board) -> None:
    """Record a position the game has reached."""
    global _last_fullmove
    # A capture or pawn move makes every earlier position unreachable, and a move counter
    # that went backwards means a new game: either way the history is worthless.
    if board.halfmove_clock == 0 or board.fullmove_number < _last_fullmove:
        _game_history.clear()
    _last_fullmove = board.fullmove_number
    _game_history.add(position_key(board))


def new_game() -> None:
    _game_history.clear()


def terminal_score(board: chess.Board, moves: list[chess.Move], ply: int) -> float | None:
    """Score of a finished game from the mover's view, or None if the game goes on.

    Checked before static evaluation and before any pruning. Mates found sooner score
    higher for the winner (``MATE_SCORE - ply``), so the engine prefers the fastest mate
    and, when lost, the slowest one.
    """
    if not moves:
        return -(MATE_SCORE - ply) if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return 0.0
    # A position can only repeat an earlier one after reversible moves, so the halfmove
    # clock is a cheap filter before the string-building key or the stack walk. The
    # board starts fresh from the FEN each move, so is_repetition only sees the current
    # search line; the game history covers everything before that.
    if board.halfmove_clock >= 2 and (
        position_key(board) in _game_history
        or (board.halfmove_clock >= 4 and board.is_repetition(2))
    ):
        return 0.0
    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move's time budget is spent."""


@dataclass
class SearchStats:
    """Instrumentation, reset before every root search."""

    nodes_searched: int = 0
    beta_cutoffs: int = 0

    def reset(self) -> None:
        self.nodes_searched = 0
        self.beta_cutoffs = 0


class Searcher:
    """One fixed-depth search from one root. Holds the counters and an optional deadline."""

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.stats = SearchStats()

    def _visit(self) -> None:
        self.stats.nodes_searched += 1
        # time.monotonic is cheap but not free: poll it every 256 nodes.
        if (
            self.deadline is not None
            and self.stats.nodes_searched & 255 == 0
            and time.monotonic() > self.deadline
        ):
            raise OutOfTime

    # -- 01: plain Negamax, kept as the reference the pruned search is checked against --

    def negamax_plain(self, board: chess.Board, depth: int, ply: int = 0) -> float:
        """Best score the side to move can force within ``depth`` plies. No pruning."""
        self._visit()
        moves = list(board.legal_moves)
        finished = terminal_score(board, moves, ply)
        if finished is not None:
            return finished
        if depth == 0:
            return evaluate(board)

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax_plain(board, depth - 1, ply + 1)
            board.pop()
            best = max(best, score)
        return best

    # -- 02: the same search with an alpha-beta window --

    def negamax(
        self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0
    ) -> float:
        """Negamax with alpha-beta pruning.

        ``alpha`` is the score the side to move is already guaranteed elsewhere in the
        tree, ``beta`` the score above which the opponent would never let us reach this
        node. Once a move proves the node is worth at least ``beta`` there is no point
        looking at the rest: the opponent will avoid this node anyway.

        Fail-soft: the value returned may lie outside [alpha, beta], but it is exact
        whenever the true value lies inside the window, which is all the root needs.
        """
        self._visit()
        moves = list(board.legal_moves)
        finished = terminal_score(board, moves, ply)
        if finished is not None:
            return finished
        if depth == 0:
            return evaluate(board)

        best = -INFINITY
        for move in moves:
            board.push(move)
            # The child sees the window from the other side, so it is negated and swapped.
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()  # always restore the board before deciding to stop

            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                self.stats.beta_cutoffs += 1
                break
        return best

    def search_root(
        self, board: chess.Board, depth: int, pruning: bool = True
    ) -> tuple[chess.Move | None, float]:
        """Search every root move with a full window; return the best move and its score.

        ``pruning=False`` runs the plain Negamax of step 01 instead, for comparison.
        Returns ``(None, score)`` when the root has no legal moves.
        """
        self.stats.reset()
        moves = list(board.legal_moves)
        finished = terminal_score(board, moves, 0)
        if not moves:
            # finished is never None here: no moves means mate or stalemate
            return None, finished if finished is not None else 0.0

        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY

        for move in moves:
            board.push(move)
            try:
                if pruning:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.negamax_plain(board, depth - 1, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            # No cutoff test at the root: beta stays +INF, and every move must be scored
            # so the best one is known.
        return best_move, best_score


# ---------------------------------------------------------------------------
# Time management
# ---------------------------------------------------------------------------
#
# This step has no iterative deepening, so the depth is chosen from the clock up front.
# A deadline still guards the search: if the chosen depth turns out too slow, the search
# aborts and the best root move scored so far (or the depth-1 answer) is played.

MOVES_TO_GO = 30
MIN_BUDGET_MS = 40
MAX_BUDGET_MS = 6_000
# (minimum budget in ms, depth to search). Checked from the deepest down. Unordered
# alpha-beta in Python runs at roughly 10k nodes a second: depth 3 costs up to ~0.7 s on
# a busy middlegame and depth 4 several seconds, hence the wide margins.
DEPTH_SCHEDULE: tuple[tuple[int, int], ...] = (
    (5_000, 4),
    (250, 3),
    (60, 2),
)


def move_budget_ms(time_left_ms: int) -> int:
    """Spend a slice of what is left on this move, never all of it."""
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, time_left_ms // MOVES_TO_GO))


def choose_depth(budget_ms: int) -> int:
    for minimum_ms, depth in DEPTH_SCHEDULE:
        if budget_ms >= minimum_ms:
            return depth
    return 1


def search_root_guarded(
    searcher: Searcher, board: chess.Board, depth: int
) -> chess.Move | None:
    """``search_root`` under the searcher's deadline; None if time ran out first.

    OutOfTime is raised somewhere deep in the tree, past frames that have pushed moves
    and not yet popped them, so the board is unwound back to the root here.
    """
    root_plies = len(board.move_stack)
    try:
        move, _ = searcher.search_root(board, depth)
    except OutOfTime:
        while len(board.move_stack) > root_plies:
            board.pop()
        return None
    return move


def choose_move(board: chess.Board, time_left_ms: int) -> chess.Move:
    budget_ms = move_budget_ms(time_left_ms)
    depth = choose_depth(budget_ms)
    deadline = time.monotonic() + budget_ms / 1000.0

    # Depth 1 is a few dozen evaluations and always finishes: the answer of last resort.
    best, _ = Searcher().search_root(board, 1)
    assert best is not None, "choose_move called with no legal moves"
    if depth == 1:
        return best

    # Without move ordering the target depth can blow the budget on a busy position, so
    # from depth 3 up the previous depth is completed first as a full-strength fallback.
    steps = [depth] if depth == 2 else [depth - 1, depth]
    for step in steps:
        searcher = Searcher(deadline)
        started = time.monotonic()
        move = search_root_guarded(searcher, board, step)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        print(
            f"depth {step}  {'aborted, playing ' if move is None else 'best '}"
            f"{(move or best).uci()}  nodes {searcher.stats.nodes_searched}  "
            f"cutoffs {searcher.stats.beta_cutoffs}  {elapsed_ms:.0f} ms of {budget_ms} ms"
        )
        if move is None:
            break
        best = move
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    remember(board)
    move = choose_move(board, time_left_ms)
    board.push(move)
    remember(board)  # the position the opponent will see
    return move.uci()
