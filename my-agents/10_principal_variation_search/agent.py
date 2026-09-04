"""Principal Variation Search chess agent.

Implements my-agents-readmes/10_principal_variation_search.md on top of the
prerequisites that spec assumes are already in place:

- negamax with alpha-beta pruning (fail-soft)
- material + piece-square evaluation from the mover's point of view
- quiescence search at the leaves
- iterative deepening with a wall-clock budget
- move ordering: TT move, promotions, MVV-LVA captures, killers, history
- transposition table keyed by python-chess's transposition key

PVS itself lives in `Searcher.negamax` and `Searcher.search_root`: the first
(best-ordered) move is searched with the full (alpha, beta) window and every
later move with a null window (alpha, alpha + 1). A null-window search can only
tell us "this move is worse than alpha" or "this move is better than alpha";
only in the second case do we pay for a full-window re-search to find out by how
much. Scores are integers so the null window is exactly one centipawn wide.

Every optional feature is a flag on `Config` so PVS can be switched off and
compared against plain alpha-beta (see test_pvs.py in this directory).

Interface required by the platform: `get_move(fen, time_left_ms) -> str` (UCI).
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables as drawn on a diagram with White at the bottom (rank 8
# first). `_from_diagram` flips them so index 0 is a1, matching python-chess.
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
_KING_MIDDLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
_KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]


def _from_diagram(table: list[int]) -> list[int]:
    """Reorder a diagram-style table (rank 8 first) into a1..h8 square order."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


def _build_tables(king: list[int]) -> dict[chess.Color, dict[chess.PieceType, list[int]]]:
    """PST[colour][piece][square] = material + square bonus for that piece there."""
    white: dict[chess.PieceType, list[int]] = {}
    for piece, diagram in (
        (chess.PAWN, _PAWN),
        (chess.KNIGHT, _KNIGHT),
        (chess.BISHOP, _BISHOP),
        (chess.ROOK, _ROOK),
        (chess.QUEEN, _QUEEN),
        (chess.KING, king),
    ):
        white[piece] = [PIECE_VALUE[piece] + bonus for bonus in _from_diagram(diagram)]
    black = {
        piece: [table[chess.square_mirror(square)] for square in chess.SQUARES]
        for piece, table in white.items()
    }
    return {chess.WHITE: white, chess.BLACK: black}


PST_MIDDLE = _build_tables(_KING_MIDDLE)
PST_END = _build_tables(_KING_END)

# Once the non-pawn material on the board drops to this, the king walks to the centre.
ENDGAME_MATERIAL = 2 * PIECE_VALUE[chess.ROOK] + 2 * PIECE_VALUE[chess.KNIGHT]


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns from the perspective of the side to move."""
    non_pawn = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        non_pawn += PIECE_VALUE[piece_type] * chess.popcount(board.pieces_mask(piece_type, True))
        non_pawn += PIECE_VALUE[piece_type] * chess.popcount(board.pieces_mask(piece_type, False))
    tables = PST_END if non_pawn <= ENDGAME_MATERIAL else PST_MIDDLE

    mover = board.turn
    score = 0
    for colour, sign in ((mover, 1), (not mover, -1)):
        own = board.occupied_co[colour]
        colour_tables = tables[colour]
        for piece_type, mask in (
            (chess.PAWN, board.pawns),
            (chess.KNIGHT, board.knights),
            (chess.BISHOP, board.bishops),
            (chess.ROOK, board.rooks),
            (chess.QUEEN, board.queens),
            (chess.KING, board.kings),
        ):
            table = colour_tables[piece_type]
            score += sign * sum(table[square] for square in chess.scan_forward(mask & own))
    return score


# ---------------------------------------------------------------------------
# Scores and constants
# ---------------------------------------------------------------------------

MATE_SCORE = 100_000  # "mate in 0"; a mate delivered at ply p scores MATE_SCORE - p
MATE_BOUND = MATE_SCORE - 1_000  # anything beyond this magnitude is a mate score
INFINITY = 1_000_000  # strictly wider than any real score, so windows stay finite
MAX_PLY = 128

TT_EXACT = 0
TT_LOWER = 1  # score is a lower bound: the node failed high
TT_UPPER = 2  # score is an upper bound: the node failed low


@dataclass(frozen=True)
class Config:
    """Feature switches. Every one can be turned off to compare against plain alpha-beta."""

    use_pvs: bool = True
    use_tt: bool = True
    use_killers: bool = True
    use_history: bool = True
    use_ordering: bool = True  # False = python-chess generation order, deliberately poor


DEFAULT_CONFIG = Config()


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------


class TTEntry:
    __slots__ = ("depth", "flag", "move", "score")

    def __init__(self, depth: int, score: int, flag: int, move: chess.Move | None) -> None:
        self.depth = depth
        self.score = score
        self.flag = flag
        self.move = move


TT_MAX_ENTRIES = 1_000_000  # a few hundred MB of Python objects at most

# python-chess keeps a tuple of the bitboards plus side to move, castling and
# en-passant state for exactly this purpose. It is ~25x cheaper to obtain than a
# polyglot Zobrist hash and distinguishes everything the legal-move state depends on.
PositionKey = Hashable


def position_key(board: chess.Board) -> PositionKey:
    return board._transposition_key()


class TranspositionTable:
    """Dictionary keyed by position. Deeper entries win; equal depth is replaced."""

    def __init__(self) -> None:
        self.table: dict[PositionKey, TTEntry] = {}
        self.hits = 0

    def probe(self, key: PositionKey) -> TTEntry | None:
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def store(
        self, key: PositionKey, depth: int, score: int, flag: int, move: chess.Move | None
    ) -> None:
        existing = self.table.get(key)
        if existing is not None and existing.depth > depth:
            return
        if len(self.table) >= TT_MAX_ENTRIES:
            self.table.clear()
        self.table[key] = TTEntry(depth, score, flag, move)

    def clear(self) -> None:
        self.table.clear()
        self.hits = 0


def score_to_tt(score: int, ply: int) -> int:
    """Mate scores are stored relative to the node, not to the root."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

ORDER_TT = 10_000_000
ORDER_PROMOTION = 9_000_000
ORDER_CAPTURE = 8_000_000
ORDER_KILLER_0 = 7_000_000
ORDER_KILLER_1 = 6_900_000
HISTORY_CAP = 5_000_000  # history values are halved when any one reaches this


class Searcher:
    """One search from one root position. Holds the deadline, tables and counters."""

    def __init__(
        self,
        deadline: float = float("inf"),
        config: Config = DEFAULT_CONFIG,
        tt: TranspositionTable | None = None,
    ) -> None:
        self.deadline = deadline
        self.config = config
        self.tt = tt if tt is not None else TranspositionTable()
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
        # history[colour * 4096 + from * 64 + to]
        self.history: list[int] = [0] * (2 * 64 * 64)
        # positions already seen in the game (outside the search tree); a repeat is a draw
        self.game_history: set[PositionKey] = set()
        self.nodes = 0
        self.qnodes = 0
        self.null_window_searches = 0
        self.researches = 0
        self.pv: list[chess.Move] = []

    # -- bookkeeping ------------------------------------------------------

    def _tick(self) -> None:
        # Check the clock every 256 nodes: Python manages tens of thousands of nodes a
        # second, so a coarser check can overshoot a small budget by a whole move's worth.
        self.nodes += 1
        if self.nodes & 255 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    @staticmethod
    def _history_index(board: chess.Board, move: chess.Move) -> int:
        return (int(board.turn) << 12) | (move.from_square << 6) | move.to_square

    def _record_cutoff(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        """A quiet move refuted the line: remember it as killer and bump its history."""
        if board.is_capture(move) or move.promotion:
            return
        if self.config.use_killers:
            slot = self.killers[ply]
            if slot[0] != move:
                slot[1] = slot[0]
                slot[0] = move
        if self.config.use_history:
            index = self._history_index(board, move)
            self.history[index] += depth * depth
            if self.history[index] >= HISTORY_CAP:
                self.history = [value >> 1 for value in self.history]

    def _priority(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return ORDER_TT
        if move.promotion:
            return ORDER_PROMOTION + PIECE_VALUE[move.promotion]
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]  # en passant
            attacker = board.piece_type_at(move.from_square)
            attacker_value = PIECE_VALUE[attacker] if attacker else 0
            return ORDER_CAPTURE + 10 * victim_value - attacker_value
        if self.config.use_killers:
            slot = self.killers[ply]
            if move == slot[0]:
                return ORDER_KILLER_0
            if move == slot[1]:
                return ORDER_KILLER_1
        if self.config.use_history:
            return self.history[self._history_index(board, move)]
        return 0

    def ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        if not self.config.use_ordering:
            return moves
        moves.sort(key=lambda move: self._priority(board, move, tt_move, ply), reverse=True)
        return moves

    def tactical_moves(self, board: chess.Board, ply: int) -> list[chess.Move]:
        """Captures and promotions only, best victim first. Much cheaper than all legal moves."""
        moves = list(board.generate_legal_captures())
        seventh = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
        promoting = board.pawns & board.occupied_co[board.turn] & seventh
        if promoting:
            moves.extend(board.generate_legal_moves(promoting, ~board.occupied))
        if self.config.use_ordering:
            moves.sort(key=lambda move: self._priority(board, move, None, ply), reverse=True)
        return moves

    # -- quiescence -------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Resolve captures and promotions so the leaf evaluation is not mid-exchange."""
        self._tick()
        self.qnodes += 1
        # A static score means nothing when the mover is mated; is_check is cheap,
        # so only pay for move generation in that case.
        if board.is_check() and not any(board.legal_moves):
            return -(MATE_SCORE - ply)
        stand_pat = evaluate(board)
        if stand_pat >= beta or ply >= MAX_PLY:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        best = stand_pat
        for move in self.tactical_moves(board, ply):
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

    # -- principal variation search ---------------------------------------

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Best score the side to move can force within `depth` plies, fail-soft.

        With `use_pvs` the first move gets the full window and later moves a null
        window; a null-window result inside (alpha, beta) triggers a re-search.
        """
        self._tick()

        if ply > 0:
            if board.halfmove_clock >= 100 or board.is_repetition(2):
                return 0
            if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
                return 0
            # Mate-distance pruning: no line from here can beat a mate already found.
            alpha = max(alpha, -MATE_SCORE + ply)
            beta = min(beta, MATE_SCORE - ply - 1)
            if alpha >= beta:
                return alpha

        key = position_key(board)
        if ply > 0 and key in self.game_history:
            return 0

        tt_move: chess.Move | None = None
        if self.config.use_tt:
            entry = self.tt.probe(key)
            if entry is not None:
                tt_move = entry.move
                if entry.depth >= depth and ply > 0:
                    score = score_from_tt(entry.score, ply)
                    if entry.flag == TT_EXACT:
                        return score
                    if entry.flag == TT_LOWER and score > alpha:
                        alpha = score
                    elif entry.flag == TT_UPPER and score < beta:
                        beta = score
                    if alpha >= beta:
                        return score

        if depth <= 0 or ply >= MAX_PLY:
            return self.quiescence(board, alpha, beta, ply)

        moves = self.ordered_moves(board, tt_move, ply)
        if not moves:
            return -(MATE_SCORE - ply) if board.is_check() else 0

        original_alpha = alpha
        best = -INFINITY
        best_move: chess.Move | None = None
        use_pvs = self.config.use_pvs

        for index, move in enumerate(moves):
            board.push(move)
            if index == 0 or not use_pvs:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Null window: can this move beat alpha at all?
                self.null_window_searches += 1
                score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    # It can, and we have no exact score for it yet: re-search fully.
                    self.researches += 1
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        self._record_cutoff(board, move, depth, ply)
                        break

        if self.config.use_tt:
            if best <= original_alpha:
                flag = TT_UPPER
            elif best >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self.tt.store(key, depth, score_to_tt(best, ply), flag, best_move)
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, int]:
        """PVS at the root. Returns the best move and its score."""
        key = position_key(board)
        tt_move: chess.Move | None = None
        if self.config.use_tt:
            entry = self.tt.probe(key)
            if entry is not None:
                tt_move = entry.move
        moves = self.ordered_moves(board, tt_move, 0)
        if not moves:
            raise ValueError("search_root called with no legal moves")

        alpha, beta = -INFINITY, INFINITY
        best_move = moves[0]
        best = -INFINITY
        use_pvs = self.config.use_pvs

        for index, move in enumerate(moves):
            board.push(move)
            if index == 0 or not use_pvs:
                score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                self.null_window_searches += 1
                score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    self.researches += 1
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score

        if self.config.use_tt:
            self.tt.store(key, depth, score_to_tt(best, 0), TT_EXACT, best_move)
        self.pv = self.principal_variation(board, best_move)
        return best_move, best

    def principal_variation(self, board: chess.Board, first: chess.Move) -> list[chess.Move]:
        """Follow TT best moves from the root to reconstruct the expected line."""
        line = [first]
        if not self.config.use_tt:
            return line
        board.push(first)
        seen = {position_key(board)}
        try:
            while len(line) < MAX_PLY:
                entry = self.tt.probe(position_key(board))
                if entry is None or entry.move is None or entry.move not in board.legal_moves:
                    break
                line.append(entry.move)
                board.push(entry.move)
                key = position_key(board)
                if key in seen:
                    break
                seen.add(key)
        finally:
            for _ in range(len(line)):
                board.pop()
        return line


def search_fixed_depth(
    board: chess.Board, depth: int, config: Config = DEFAULT_CONFIG
) -> tuple[chess.Move, int, Searcher]:
    """Single fixed-depth search with no deadline. Used by tests and benchmarks."""
    searcher = Searcher(config=config)
    move, score = searcher.search_root(board, depth)
    return move, score, searcher


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30
MIN_BUDGET_MS = 30
MAX_BUDGET_MS = 8_000


def move_budget_ms(time_left_ms: int) -> int:
    """Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    config: Config = DEFAULT_CONFIG,
    tt: TranspositionTable | None = None,
    game_history: set[PositionKey] | None = None,
) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth."""
    started = time.monotonic()
    budget_s = move_budget_ms(time_left_ms) / 1000.0
    root_stack = len(board.move_stack)
    searcher = Searcher(config=config, tt=tt)
    if game_history is not None:
        searcher.game_history = game_history

    # Depth 1 runs without a deadline so there is always something to play, even
    # when the budget is tiny and quiescence is busy.
    best_move, best_score = searcher.search_root(board, 1)
    searcher.deadline = started + budget_s

    for depth in range(2, MAX_DEPTH + 1):
        # An iteration usually costs several times the previous one: do not start
        # one that has no realistic chance of finishing.
        elapsed = time.monotonic() - started
        if elapsed > budget_s * 0.4:
            break
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            # The exception skipped the pops on the way up: unwind the board.
            while len(board.move_stack) > root_stack:
                board.pop()
            break
        pv = " ".join(move.uci() for move in searcher.pv[:6])
        print(
            f"depth {depth:2d}  score {best_score:7d}  nodes {searcher.nodes:8d}  "
            f"nw {searcher.null_window_searches:7d}  re {searcher.researches:6d}  pv {pv}"
        )
        if abs(best_score) >= MATE_BOUND:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


# Module state survives between moves of one game (the process is per game).
_TT = TranspositionTable()
_GAME_HISTORY: set[PositionKey] = set()


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    move = choose_move(board, time_left_ms, tt=_TT, game_history=_GAME_HISTORY)
    _GAME_HISTORY.add(position_key(board))
    board.push(move)
    _GAME_HISTORY.add(position_key(board))
    return move.uci()
