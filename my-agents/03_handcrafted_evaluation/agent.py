"""Chess agent for stage 03: Negamax + alpha-beta driven by a handcrafted evaluation.

The evaluation lives in evaluation.py next to this file (see
my-agents-readmes/03_handcrafted_evaluation.md). This module only supplies the search
from stages 01 and 02, plus iterative deepening on a time budget so the agent never
loses on the clock:

    negamax(board, depth, alpha, beta)   one recursive function, child scores negated
    search_root(board, depth)            same, but remembers the best move
    choose_move(board, time_left_ms)     deepen until the budget for this move runs out

Lives at my-agents/03_handcrafted_evaluation/agent.py so the harness can import it. The
harness puts this directory first on sys.path, which is how `import evaluation` works.
"""

from __future__ import annotations

import math
import time

import chess
from evaluation import PIECE_VALUE, evaluate

MATE_SCORE = 100_000  # bigger than any material swing
INFINITY = math.inf

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 50
MAX_BUDGET_MS = 8_000

_BB_SQUARES = chess.BB_SQUARES


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    """Legal moves, captures and promotions first.

    Alpha-beta prunes most when the best move is searched first. Trying "take the
    queen with a pawn" before quiet moves is a cheap approximation of that: most
    valuable victim, least valuable attacker. Bitboard tests are used instead of
    board.is_capture so ordering stays cheap.
    """
    enemy = board.occupied_co[not board.turn]
    ep_square = board.ep_square
    piece_type_at = board.piece_type_at

    def priority(move: chess.Move) -> int:
        score = 0
        if move.promotion:
            score += 10_000 + PIECE_VALUE[move.promotion]
        to_bb = _BB_SQUARES[move.to_square]
        if enemy & to_bb:
            victim = piece_type_at(move.to_square) or chess.PAWN
            attacker = piece_type_at(move.from_square) or chess.PAWN
            score += 1_000 + 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
        elif move.to_square == ep_square and piece_type_at(move.from_square) == chess.PAWN:
            score += 1_000 + 10 * PIECE_VALUE[chess.PAWN] - PIECE_VALUE[chess.PAWN]
        return score

    moves = list(board.legal_moves)
    moves.sort(key=priority, reverse=True)
    return moves


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search for one root position. Holds the deadline and node counters."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.beta_cutoffs = 0

    def _tick(self) -> None:
        # time.monotonic is cheap, but not free; check every 256 nodes.
        self.nodes += 1
        if self.nodes & 255 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Best score the side to move can force within `depth` plies."""
        self._tick()

        # Draw by repetition or the fifty-move rule, checked before generating moves so
        # we never "win" material in a line the opponent can simply repeat out of.
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0.0

        if depth == 0:
            # Leaf. Generating every legal move here only to learn whether the side to
            # move is mated is the single most expensive thing a shallow search does, so
            # only do it when the king is actually in check.
            if board.is_check() and not any(board.generate_legal_moves()):
                return -float(MATE_SCORE)
            return float(evaluate(board))

        moves = ordered_moves(board)
        if not moves:
            # Checkmated: losing, and sooner is worse than later, so subtract depth.
            # Stalemate: draw.
            return -(MATE_SCORE + depth) if board.is_check() else 0.0

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                self.beta_cutoffs += 1
                break  # the opponent will never allow this line; stop looking
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        """Negamax at the root, remembering which move produced the best score."""
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY

        for move in ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        assert best_move is not None, "search_root called with no legal moves"
        return best_move, best_score


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(board: chess.Board, time_left_ms: int) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth.

    Each completed depth was a full search, so the move from the deepest finished
    iteration is always a sound answer. Depth 1 is never interrupted, so there is
    always a legal move to play.
    """
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(deadline)

    best_move, best_score = searcher.search_root(board, 1)

    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        print(
            f"depth {depth:2d}  score {best_score:8.0f}  best {best_move.uci()}  "
            f"nodes {searcher.nodes}  cutoffs {searcher.beta_cutoffs}"
        )
        if abs(best_score) >= MATE_SCORE:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
