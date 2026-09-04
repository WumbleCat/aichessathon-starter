"""Killer-heuristic chess agent.

Implements my-agents-readmes/08_killer_heuristic.md on top of a plain alpha-beta
searcher:

1. evaluation            material plus piece-square tables, from the mover's view
2. negamax + alpha-beta  one maximising function, prune what cannot change the root
3. quiescence            resolve captures at the leaves
4. iterative deepening   depth 1, 2, 3 ... until the move budget runs out
5. move ordering         hash move, promotions, captures (MVV-LVA), then quiet moves
6. killer heuristic      two quiet moves per ply that recently caused a beta cutoff
                         are tried right after the captures      <- this file's subject

The killer table is indexed by ply, not by position: a quiet move that refuted one
line is a good guess for refuting a sibling line at the same distance from the root.
It only changes the order in which legal moves are tried, never which moves exist,
so with USE_KILLERS on or off the search returns the same score at the same depth.

Lives at my-agents/08_killer_heuristic/agent.py so the harness can import it. To submit,
copy it to agent.py at the root of the repo.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_KILLERS = True  # the feature under test; flip to compare node counts
USE_HASH_MOVE = True  # remember the best move per position and try it first

MAX_PLY = 128  # deepest ply the killer table covers (search never gets near this)
KILLER_SLOTS = 2

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

# Piece-square tables written as a diagram with White at the bottom (rank 8 first),
# flipped below so index 0 is a1 to match python-chess square numbering.
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

MATE_SCORE = 100_000  # bigger than any material swing
INFINITY = math.inf


_PIECE_BITBOARDS: tuple[tuple[chess.PieceType, str], ...] = (
    (chess.PAWN, "pawns"),
    (chess.KNIGHT, "knights"),
    (chess.BISHOP, "bishops"),
    (chess.ROOK, "rooks"),
    (chess.QUEEN, "queens"),
    (chess.KING, "kings"),
)


def evaluate(board: chess.Board) -> int:
    """Static score from the side to move's point of view. Positive: the mover is better.

    Walks the bitboards directly; board.piece_map() builds a Piece object per square
    and was the single most expensive call in the profile.
    """
    score = 0
    for piece_type, attribute in _PIECE_BITBOARDS:
        pieces: int = getattr(board, attribute)
        value = PIECE_VALUE[piece_type]
        for colour in (chess.WHITE, chess.BLACK):
            table = PST[colour][piece_type]
            total = 0
            for square in chess.scan_forward(pieces & board.occupied_co[colour]):
                total += value + table[square]
            score += total if colour == board.turn else -total
    return score


# ---------------------------------------------------------------------------
# 6. Killer heuristic
# ---------------------------------------------------------------------------


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    """A move that is neither a capture nor a promotion. Only these become killers.

    Captures and promotions are already ordered well by MVV-LVA / promotion bonuses,
    and their value depends on the material on the board, which differs between
    sibling positions. A quiet move's refuting power transfers between siblings much
    more often, which is the whole bet the killer heuristic makes.
    """
    return not board.is_capture(move) and move.promotion is None


class KillerTable:
    """Two quiet moves per ply that caused a beta cutoff there, most recent first."""

    def __init__(self, max_ply: int = MAX_PLY, slots: int = KILLER_SLOTS) -> None:
        self.slots = slots
        self.table: list[list[chess.Move | None]] = [[None] * slots for _ in range(max_ply)]

    def clear(self) -> None:
        for entry in self.table:
            for i in range(self.slots):
                entry[i] = None

    def get(self, ply: int) -> list[chess.Move | None]:
        return self.table[ply] if ply < len(self.table) else [None] * self.slots

    def store(self, ply: int, move: chess.Move) -> bool:
        """Record a killer at this ply. Returns True if the table changed.

        The new move goes into slot 0 and the previous slot 0 shifts to slot 1. If the
        move is already the primary killer nothing happens, so the same move can never
        occupy both slots.
        """
        if ply >= len(self.table):
            return False
        entry = self.table[ply]
        if entry[0] == move:
            return False
        for i in range(self.slots - 1, 0, -1):
            entry[i] = entry[i - 1]
        entry[0] = move
        return True

    def rank(self, ply: int, move: chess.Move) -> int:
        """0 for the primary killer, 1 for the secondary, ..., -1 if not a killer."""
        if ply >= len(self.table):
            return -1
        entry = self.table[ply]
        for i in range(self.slots):
            if entry[i] == move:
                return i
        return -1


# ---------------------------------------------------------------------------
# 5. Move ordering
# ---------------------------------------------------------------------------

HASH_MOVE_BONUS = 1_000_000
PROMOTION_BONUS = 100_000
CAPTURE_BONUS = 10_000
KILLER_BONUS = 9_000  # just below every capture, above every other quiet move
KILLER_STEP = 1_000  # killer 0 outranks killer 1


def move_order_score(
    board: chess.Board,
    move: chess.Move,
    hash_move: chess.Move | None,
    killers: KillerTable | None,
    ply: int,
) -> int:
    """Higher is searched earlier.

    hash move > promotions > captures (MVV-LVA) > killer 0 > killer 1 > other quiet moves
    """
    if move == hash_move:
        return HASH_MOVE_BONUS
    score = 0
    if move.promotion is not None:
        score += PROMOTION_BONUS + PIECE_VALUE[move.promotion]
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        # An en passant capture lands on an empty square; the victim is a pawn.
        victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker else 0
        score += CAPTURE_BONUS + 10 * victim_value - attacker_value
    elif score == 0 and killers is not None:
        # A quiet move: does it match one of the killers at this ply?
        rank = killers.rank(ply, move)
        if rank >= 0:
            score += KILLER_BONUS - KILLER_STEP * rank
    return score


def ordered_moves(
    board: chess.Board,
    hash_move: chess.Move | None = None,
    killers: KillerTable | None = None,
    ply: int = 0,
) -> list[chess.Move]:
    """Every legal move, best guess first. Nothing is dropped, only reordered."""
    moves = list(board.legal_moves)
    moves.sort(
        key=lambda move: move_order_score(board, move, hash_move, killers, ply), reverse=True
    )
    return moves


def tactical_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and promotions only, MVV-LVA first. What quiescence searches.

    Generating just these is far cheaper than generating every legal move and then
    throwing the quiet ones away.
    """
    moves = list(board.generate_legal_captures())
    pawns = board.pawns & board.occupied_co[board.turn]
    promoting = pawns & (chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2)
    if promoting:
        moves.extend(
            move
            for move in board.generate_legal_moves(promoting, ~board.occupied)
            if move.promotion is not None
        )
    moves.sort(key=lambda move: move_order_score(board, move, None, None, 0), reverse=True)
    return moves


# ---------------------------------------------------------------------------
# 2 + 3. Negamax with alpha-beta and quiescence
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


@dataclass
class SearchStats:
    nodes: int = 0
    qnodes: int = 0
    cutoffs: int = 0
    first_move_cutoffs: int = 0  # cutoffs produced by the first move tried
    killer_cutoffs: int = 0  # cutoffs produced by a move that was a killer
    killers_stored: int = 0
    depth: int = 0
    best_move: str = ""
    score: float = 0.0
    elapsed_s: float = 0.0


class Searcher:
    """One search for one root position. Holds the deadline, the tables and counters."""

    def __init__(
        self,
        deadline: float = INFINITY,
        use_killers: bool = USE_KILLERS,
        use_hash_move: bool = USE_HASH_MOVE,
    ) -> None:
        self.deadline = deadline
        self.use_killers = use_killers
        self.use_hash_move = use_hash_move
        self.killers: KillerTable | None = KillerTable() if use_killers else None
        self.hash_moves: dict[object, chess.Move] = {}
        self.stats = SearchStats()

    # -- helpers ---------------------------------------------------------------

    def _tick(self) -> None:
        # time.monotonic is cheap but not free; check every 256 nodes. At a few thousand
        # nodes a second that bounds the overshoot past the deadline to tens of ms.
        self.stats.nodes += 1
        if self.stats.nodes & 255 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    @staticmethod
    def _key(board: chess.Board) -> object:
        # python-chess's own repetition key: a tuple of bitboards plus castling, en
        # passant and side to move. Much cheaper than a Zobrist hash in pure Python.
        return board._transposition_key()

    def _hash_move(self, board: chess.Board) -> chess.Move | None:
        if not self.use_hash_move:
            return None
        return self.hash_moves.get(self._key(board))

    def _remember(self, board: chess.Board, move: chess.Move) -> None:
        if self.use_hash_move:
            self.hash_moves[self._key(board)] = move

    def _on_cutoff(self, board: chess.Board, move: chess.Move, ply: int, index: int) -> None:
        """Bookkeeping when `move` caused a beta cutoff at `ply` as the `index`-th move tried.

        This is where the killer heuristic stores: quiet moves only, never captures
        or promotions.
        """
        self.stats.cutoffs += 1
        if index == 0:
            self.stats.first_move_cutoffs += 1
        if self.killers is None:
            return
        if self.killers.rank(ply, move) >= 0:
            self.stats.killer_cutoffs += 1
        if is_quiet(board, move) and self.killers.store(ply, move):
            self.stats.killers_stored += 1

    # -- search ----------------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Search captures and promotions until the position is quiet."""
        self._tick()
        self.stats.qnodes += 1
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        for move in tactical_moves(board):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(
        self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int
    ) -> float:
        """Best score the side to move can force within `depth` plies."""
        self._tick()

        # Draw by repetition or the fifty-move rule. A repetition needs at least four
        # reversible plies, so the halfmove clock is a cheap guard on the expensive check.
        if board.halfmove_clock >= 100 or (
            board.halfmove_clock >= 4 and board.is_repetition(2)
        ):
            return 0.0

        moves = ordered_moves(board, self._hash_move(board), self.killers, ply)
        if not moves:
            # Checkmated: losing, and sooner is worse than later. Stalemate: draw.
            return -(MATE_SCORE - ply) if board.is_check() else 0.0

        if depth == 0:
            return self.quiescence(board, alpha, beta)

        best = -INFINITY
        best_move = moves[0]
        for index, move in enumerate(moves):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                self._on_cutoff(board, move, ply, index)
                break  # the opponent will never allow this line; stop looking
        self._remember(board, best_move)
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        """Negamax at the root, remembering which move produced the best score."""
        moves = ordered_moves(board, self._hash_move(board), self.killers, 0)
        assert moves, "search_root called with no legal moves"

        best_move = moves[0]
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        self._remember(board, best_move)
        return best_move, best_score

    def search_fixed_depth(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        """Iterative deepening to exactly `depth`, no clock. Used by tests and benchmarks."""
        started = time.monotonic()
        best_move, best_score = self.search_root(board, 1)
        for d in range(2, depth + 1):
            best_move, best_score = self.search_root(board, d)
        self.stats.depth = depth
        self.stats.best_move = best_move.uci()
        self.stats.score = best_score
        self.stats.elapsed_s = time.monotonic() - started
        return best_move, best_score


# ---------------------------------------------------------------------------
# 4. Iterative deepening and time management
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

    The killer table and the hash-move table live for the whole move: killers found at
    depth d are exactly what makes the depth d+1 iteration cheaper. They are rebuilt
    from scratch on the next call, which is the "reset between root searches" option.
    """
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(INFINITY)

    # Depth 1 runs without a clock so it always completes: there is always a move to play.
    best_move, best_score = searcher.search_root(board, 1)
    searcher.deadline = deadline

    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        stats = searcher.stats
        print(
            f"depth {depth:2d}  score {best_score:8.0f}  best {best_move.uci()}  "
            f"nodes {stats.nodes}  cutoffs {stats.cutoffs}  "
            f"first-move {stats.first_move_cutoffs}  killer {stats.killer_cutoffs}"
        )
        if abs(best_score) >= MATE_SCORE - MAX_PLY:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
