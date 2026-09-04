"""Null-move pruning chess agent.

Implements my-agents-readmes/11_null_move_pruning.md on top of a conventional
alpha-beta searcher so the pruning has a realistic tree to work on:

    evaluation            material + piece-square tables, from the mover's view
    negamax + alpha-beta  fail-soft, with mate scores measured from the root
    quiescence            captures and promotions are resolved at the leaves
    transposition table   keyed by python-chess's polyglot Zobrist hash
    move ordering         TT move, MVV-LVA captures, killer moves, history
    iterative deepening   depth 1, 2, 3 ... until the move budget runs out
    null-move pruning     the subject of this bot, see `can_try_null_move`

Null-move pruning in one paragraph: at a node where we would like to prove a
beta cutoff, first let the side to move *pass*. If, after handing the opponent
a free move, a reduced-depth search still scores >= beta, then any real move is
very likely to as well, and we return beta without searching the real moves.
It is unsound in zugzwang, where passing is better than every legal move, so it
is disabled in check, at low depth, and when the mover has little non-pawn
material (king-and-pawn endings are the classic zugzwang breeding ground).

Every safeguard and the feature itself are toggles on `SearchConfig`.

Lives at my-agents/11_null_move_pruning/agent.py so the harness can import it.
To submit, copy it to agent.py at the root of the repo (make zip puts that file
at the root of the zip).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import chess.polyglot

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

# Piece-square tables written as a diagram with White at the bottom (rank 8 first).
# They are re-indexed below so index 0 is a1, matching python-chess squares.
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

MATE_SCORE = 100_000  # "mate at the root"; a mate found at ply p scores MATE_SCORE - p
MATE_BOUND = MATE_SCORE - 1_000  # anything beyond this magnitude is a mate score
INFINITY = 10**9
DRAW_SCORE = 0
MAX_PLY = 128


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move (positive = mover better)."""
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == board.turn else -value
    return score


def non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    """Centipawns of knights, bishops, rooks and queens `color` still has on the board."""
    return sum(
        PIECE_VALUE[piece] * len(board.pieces(piece, color))
        for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    """Every selective feature is a toggle so it can be benchmarked on and off."""

    # Null-move pruning (the feature this bot is about).
    null_move: bool = True
    # Do not try a null move with fewer plies than this left. At depth 2 the reduced
    # search is pure quiescence, which is still a useful "is the pass good enough?"
    # probe. 3 is the more conservative value from the spec; measured at depth 5 on
    # four middlegames, 2 searched ~27% fewer nodes than no pruning and 3 only ~21%.
    null_move_min_depth: int = 2
    # Base reduction R. The reduced search is depth - 1 - R plies deep.
    null_move_reduction: int = 2
    # From this depth on, use R + 1. Deep nodes tolerate a bigger reduction.
    null_move_deep_depth: int = 6
    # Zugzwang guard: the side to move must own at least this much non-pawn material.
    # 500 means a rook, a queen, or two minor pieces. King-and-pawn endings are always 0.
    null_move_min_material: int = 500
    # Only try a null move when the static evaluation already stands at or above beta.
    # It rarely succeeds otherwise and the failed reduced search is wasted work.
    null_move_needs_eval_above_beta: bool = True

    # Transposition table on/off (kept as a toggle for node-count experiments).
    use_tt: bool = True


CONFIG = SearchConfig()


def null_move_reduction(depth: int, config: SearchConfig) -> int:
    """Adaptive R: the base reduction, plus one at deep nodes."""
    extra = 1 if depth >= config.null_move_deep_depth else 0
    return config.null_move_reduction + extra


def can_try_null_move(
    board: chess.Board,
    depth: int,
    beta: int,
    allow_null: bool,
    in_check: bool,
    config: SearchConfig,
    static_eval: int | None = None,
) -> bool:
    """All safeguards from the spec, in one place.

    - feature toggled off              -> no
    - the parent already passed        -> no (two passes in a row prove nothing)
    - side to move is in check         -> no (passing would leave the king en prise)
    - depth too small                  -> no
    - beta is a mate score             -> no (a pass cannot prove or refute a mate)
    - little non-pawn material         -> no (likely zugzwang, e.g. king-and-pawn)
    - static eval below beta           -> no (the pass would almost surely fail low)
    """
    if not config.null_move or not allow_null or in_check:
        return False
    if depth < config.null_move_min_depth:
        return False
    if abs(beta) >= MATE_BOUND:
        return False
    if non_pawn_material(board, board.turn) < config.null_move_min_material:
        return False
    if config.null_move_needs_eval_above_beta:
        if static_eval is None:
            static_eval = evaluate(board)
        if static_eval < beta:
            return False
    return True


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------

EXACT, LOWER_BOUND, UPPER_BOUND = 0, 1, 2

# (depth, score adjusted to be node-relative, flag, best move)
TTEntry = tuple[int, int, int, chess.Move | None]
TT_MAX_ENTRIES = 2_000_000  # a few hundred MB of Python objects at most


def score_to_tt(score: int, ply: int) -> int:
    """Mate scores are stored as "mate in N from *this* node", not from the root."""
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


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search for one root position: the deadline, counters and heuristics."""

    def __init__(
        self,
        config: SearchConfig | None = None,
        tt: dict[int, TTEntry] | None = None,
        deadline: float = float("inf"),
    ) -> None:
        self.config = config if config is not None else CONFIG
        self.tt: dict[int, TTEntry] = tt if tt is not None else {}
        self.deadline = deadline
        self.nodes = 0
        self.null_tries = 0
        self.null_cutoffs = 0
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: dict[tuple[chess.Color, chess.Square, chess.Square], int] = {}

    # -- bookkeeping ---------------------------------------------------------

    def _tick(self) -> None:
        # time.monotonic is cheap but not free; at ~7k nodes/s, 256 nodes is ~35 ms,
        # which keeps the overshoot past the deadline well inside the increment.
        self.nodes += 1
        if self.nodes & 255 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def _probe(self, key: int) -> TTEntry | None:
        return self.tt.get(key) if self.config.use_tt else None

    def _store(self, key: int, entry: TTEntry) -> None:
        if not self.config.use_tt:
            return
        if len(self.tt) >= TT_MAX_ENTRIES:
            self.tt.clear()
        self.tt[key] = entry

    def _remember_quiet_cutoff(
        self, board: chess.Board, move: chess.Move, depth: int, ply: int
    ) -> None:
        slots = self.killers[ply]
        if slots[0] != move:
            slots[1] = slots[0]
            slots[0] = move
        index = (board.turn, move.from_square, move.to_square)
        self.history[index] = self.history.get(index, 0) + depth * depth

    # -- move ordering -------------------------------------------------------

    def _priority(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return 1_000_000
        priority = 0
        if move.promotion:
            priority += 100_000 + PIECE_VALUE[move.promotion]
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
            attacker = board.piece_type_at(move.from_square)
            attacker_value = PIECE_VALUE[attacker] if attacker else 0
            return priority + 10_000 + 10 * victim_value - attacker_value
        if priority:
            return priority
        killers = self.killers[ply]
        if move == killers[0]:
            return 9_000
        if move == killers[1]:
            return 8_000
        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    def ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self._priority(board, move, tt_move, ply), reverse=True)
        return moves

    # -- quiescence ----------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Resolve captures and promotions so the leaf evaluation is not mid-exchange."""
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta or ply >= MAX_PLY - 1:
            return stand_pat
        alpha = max(alpha, stand_pat)
        best = stand_pat
        for move in self.ordered_moves(board, None, ply):
            if not board.is_capture(move) and not move.promotion:
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

    # -- the main search -----------------------------------------------------

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
    ) -> int:
        """Best score the side to move can force within `depth` plies (fail-soft)."""
        self._tick()

        # Draws are decided before anything else so no line "wins" material the
        # opponent can escape by repeating or running out the fifty-move clock.
        if ply > 0 and (
            board.is_repetition(2)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        ):
            return DRAW_SCORE
        if ply >= MAX_PLY - 1:
            return evaluate(board)

        in_check = board.is_check()
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        key = chess.polyglot.zobrist_hash(board)
        alpha_original = alpha
        tt_move: chess.Move | None = None
        entry = self._probe(key)
        if entry is not None:
            entry_depth, entry_score, entry_flag, tt_move = entry
            if entry_depth >= depth:
                score = score_from_tt(entry_score, ply)
                if entry_flag == EXACT:
                    return score
                if entry_flag == LOWER_BOUND:
                    alpha = max(alpha, score)
                elif entry_flag == UPPER_BOUND:
                    beta = min(beta, score)
                if alpha >= beta:
                    return score

        # ---- null-move pruning ------------------------------------------------
        if can_try_null_move(board, depth, beta, allow_null, in_check, self.config):
            self.null_tries += 1
            reduction = null_move_reduction(depth, self.config)
            board.push(chess.Move.null())
            try:
                # Zero window around beta: we only need to know "is it still >= beta?"
                score = -self.negamax(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, allow_null=False
                )
            finally:
                board.pop()
            if score >= beta:
                self.null_cutoffs += 1
                # Fail hard on purpose: a null-move score is a proof of a bound, not a
                # real line, and mate scores from a pass are never trusted.
                return beta
        # ------------------------------------------------------------------------

        moves = self.ordered_moves(board, tt_move, ply)
        if not moves:
            return -(MATE_SCORE - ply) if in_check else DRAW_SCORE

        best = -INFINITY
        best_move: chess.Move | None = None
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if not board.is_capture(move) and not move.promotion:
                        self._remember_quiet_cutoff(board, move, depth, ply)
                    break

        if best <= alpha_original:
            flag = UPPER_BOUND
        elif best >= beta:
            flag = LOWER_BOUND
        else:
            flag = EXACT
        self._store(key, (depth, score_to_tt(best, ply), flag, best_move))
        return best

    def search_root(
        self, board: chess.Board, depth: int, first: chess.Move | None = None
    ) -> tuple[chess.Move, int]:
        """Negamax at the root, remembering which move produced the best score."""
        key = chess.polyglot.zobrist_hash(board)
        if first is None:
            entry = self._probe(key)
            if entry is not None:
                first = entry[3]

        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        for move in self.ordered_moves(board, first, 0):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        if best_move is None:
            raise ValueError("search_root called with no legal moves")
        self._store(key, (depth, score_to_tt(best_score, 0), EXACT, best_move))
        return best_move, best_score


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 50
MAX_BUDGET_MS = 8_000

# The table survives between moves of one game (the process does), never between games.
_TRANSPOSITIONS: dict[int, TTEntry] = {}


def move_budget_ms(time_left_ms: int) -> int:
    """Spend a slice of the remaining clock, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    config: SearchConfig | None = None,
    verbose: bool = True,
) -> chess.Move:
    """Deepen one ply at a time; keep the move from the deepest finished iteration."""
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(config, _TRANSPOSITIONS, deadline)

    # Depth 1 is searched without a clock so there is always a move to play.
    searcher.deadline = float("inf")
    best_move, best_score = searcher.search_root(board, 1)
    searcher.deadline = deadline

    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth, best_move)
        except OutOfTime:
            break
        if verbose:
            print(
                f"depth {depth:2d}  score {best_score:7d}  best {best_move.uci()}  "
                f"nodes {searcher.nodes}  null {searcher.null_cutoffs}/{searcher.null_tries}"
            )
        if abs(best_score) >= MATE_BOUND:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
