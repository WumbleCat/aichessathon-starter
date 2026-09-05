"""Negamax chess agent.

Follows the progression in my-agents-readmes/negamax.md:

1. board evaluation      material plus piece-square tables, from the mover's view
2. negamax               one maximising function, the child's score is negated
3. alpha-beta pruning    skip branches that cannot change the root decision
4. move ordering         captures first, most valuable victim, least valuable attacker
5. iterative deepening   search depth 1, 2, 3 ... until the move budget runs out
6. quiescence search     at the leaves, keep resolving captures so we do not stop
                         halfway through an exchange

Lives at my-agents/negamax/agent.py so the harness can import it. To submit, copy it to
agent.py at the root of the repo (make zip puts that file at the root of the zip).
"""

import math
import time

import chess

# ---------------------------------------------------------------------------
# 1. Evaluation
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables, written as they appear on a diagram with White at the bottom
# (rank 8 first). They are flipped below so index 0 is a1, matching python-chess.
# Values are small nudges on top of material: knights like the centre, pawns like to
# advance, the king likes to stay tucked away in the middlegame.
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


# PST[colour][piece_type][square] -> bonus for that piece standing on that square.
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

MATE_SCORE = 100_000  # bigger than any material swing
INFINITY = math.inf


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move.

    Positive means the mover is better. Negamax needs exactly this convention so
    that a child's score can simply be negated.
    """
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == board.turn else -value
    return score


# ---------------------------------------------------------------------------
# 4. Move ordering
# ---------------------------------------------------------------------------


def _move_priority(board: chess.Board, move: chess.Move) -> int:
    """Higher is searched earlier. Good captures and promotions first.

    Alpha-beta prunes the most when the best move is tried first. We do not know the
    best move, but "capture the queen with a pawn" is a fine guess. This is MVV-LVA:
    most valuable victim, least valuable attacker.
    """
    priority = 0
    if move.promotion:
        priority += 10_000 + PIECE_VALUE[move.promotion]
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        # An en passant capture lands on an empty square; the victim is a pawn.
        victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker else 0
        priority += 1_000 + 10 * victim_value - attacker_value
    return priority


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: _move_priority(board, move), reverse=True)
    return moves


# ---------------------------------------------------------------------------
# 2 + 3 + 6. Negamax with alpha-beta and quiescence
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search for one root position. Holds the deadline and a node counter."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0

    def _tick(self) -> None:
        # time.monotonic is cheap, but not free; check every 2048 nodes.
        self.nodes += 1
        if self.nodes & 2047 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Search only captures until the position is quiet.

        Without this, depth-limited search stops in the middle of an exchange and
        the evaluation sees a queen "won" that is about to be taken back.
        """
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        for move in ordered_moves(board):
            if not board.is_capture(move) and not move.promotion:
                continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Best score the side to move can force within `depth` plies."""
        self._tick()

        # Draw by repetition or the fifty-move rule. Checked before generating moves
        # so we never "win" material in a line the opponent can just repeat out of.
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0.0

        moves = ordered_moves(board)
        if not moves:
            # Checkmated: losing, and sooner is worse than later so subtract depth.
            # Stalemate: draw.
            return -(MATE_SCORE + depth) if board.is_check() else 0.0

        if depth == 0:
            return self.quiescence(board, alpha, beta)

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # the opponent will never allow this line; stop looking
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        """Negamax at the root, but remember which move produced the best score."""
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


# ---------------------------------------------------------------------------
# 5. Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 50
MAX_BUDGET_MS = 8_000


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(board: chess.Board, time_left_ms: int) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth.

    The search at depth d is aborted by OutOfTime when the clock expires. Because
    each completed depth was a full search, the move from the deepest finished
    iteration is always a sound answer.
    """
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(deadline)

    # Depth 1 always completes so there is always something to play.
    best_move, best_score = searcher.search_root(board, 1)

    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        print(f"depth {depth:2d}  score {best_score:8.0f}  best {best_move.uci()}  "
              f"nodes {searcher.nodes}")
        if abs(best_score) >= MATE_SCORE:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
