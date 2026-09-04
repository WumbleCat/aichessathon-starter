"""History heuristic chess agent.

Implements my-agents-readmes/09_history_heuristic.md on top of the search stages that
come before it in the roadmap, because a history table only makes sense inside an
alpha-beta search that already has other ordering sources to slot in behind:

1. evaluation            material plus piece-square tables, from the mover's view
2. negamax + alpha-beta  one maximising function; branches that cannot change the
                         root decision are cut
3. quiescence            captures and promotions keep being resolved at the leaves
4. iterative deepening   depth 1, 2, 3 ... until the move budget runs out
5. move ordering         TT move, promotions, captures by MVV-LVA, killers, then
                         quiet moves ranked by history
6. transposition table   keyed by the polyglot Zobrist hash python-chess provides
7. killer moves          two quiet cutoff moves per ply
8. history heuristic     history[colour][from][to], depth*depth bonus on a quiet
                         beta cutoff, a matching penalty for the quiet moves tried
                         before it, halved whenever a score gets large

The history table is the point of this file. Everything else is kept as plain as
possible so the ordering code is easy to follow.

Lives at my-agents/09_history_heuristic/agent.py so the harness can import it. To
submit, copy it to agent.py at the root of the repo (make zip puts that file at the
root of the zip).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import chess.polyglot

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

# Piece-square tables written as a diagram with White at the bottom (rank 8 first).
# They are flipped below so index 0 is a1, matching python-chess square numbering.
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
# PST[colour][piece_type][square] -> bonus for that piece standing on that square.
PST: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: _WHITE_PST,
    chess.BLACK: _BLACK_PST,
}

MATE_SCORE = 100_000  # bigger than any material swing
INFINITY = MATE_SCORE * 10  # a finite "infinite" keeps every score an int


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move. Positive: mover is better."""
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == board.turn else -value
    return score


# ---------------------------------------------------------------------------
# 8. History heuristic
# ---------------------------------------------------------------------------

# Quiet moves score inside (-HISTORY_MAX, HISTORY_MAX). When any entry leaves that band
# the whole table is halved, so old cutoffs fade and nothing grows without bound.
HISTORY_MAX = 16_384


class HistoryTable:
    """history[colour][from_square][to_square] -> how often the move caused a cutoff.

    Stored as one flat list of 2 * 64 * 64 ints so a lookup is a single index.
    Captures and promotions never touch the table: they have their own ordering.
    """

    SIZE = 2 * 64 * 64

    def __init__(self) -> None:
        self.table: list[int] = [0] * self.SIZE

    @staticmethod
    def index(colour: chess.Color, move: chess.Move) -> int:
        return (int(colour) << 12) | (move.from_square << 6) | move.to_square

    def get(self, colour: chess.Color, move: chess.Move) -> int:
        return self.table[self.index(colour, move)]

    def reset(self) -> None:
        self.table = [0] * self.SIZE

    def age(self) -> None:
        """Halve everything. Called between root searches so stale cutoffs fade."""
        self.table = [value >> 1 if value >= 0 else -((-value) >> 1) for value in self.table]

    def update(
        self, colour: chess.Color, cutoff: chess.Move, tried: list[chess.Move], depth: int
    ) -> None:
        """A quiet `cutoff` move just refuted a line at `depth`.

        Reward it with depth*depth and penalise, by the same amount, the quiet moves
        that were searched before it and failed to cut. If the table gets big, age it.
        """
        bonus = depth * depth
        table = self.table
        table[self.index(colour, cutoff)] += bonus
        for move in tried:
            table[self.index(colour, move)] -= bonus
        touched = [cutoff, *tried]
        if any(abs(table[self.index(colour, move)]) >= HISTORY_MAX for move in touched):
            self.age()


# ---------------------------------------------------------------------------
# 6. Transposition table
# ---------------------------------------------------------------------------

EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2


@dataclass(slots=True)
class TTEntry:
    key: int
    depth: int
    score: int  # mate scores are stored relative to this node, see _to_tt / _from_tt
    flag: int
    best_move: chess.Move | None


TT_MAX_ENTRIES = 1_000_000  # a few hundred MB worst case, well under the 2 GB cap


def _to_tt(score: int, ply: int) -> int:
    """Mate scores are "mate in N from the root"; store them as "from this node"."""
    if score >= MATE_SCORE - 1000:
        return score + ply
    if score <= -MATE_SCORE + 1000:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 1000:
        return score - ply
    if score <= -MATE_SCORE + 1000:
        return score + ply
    return score


# ---------------------------------------------------------------------------
# 5. Move ordering
# ---------------------------------------------------------------------------

# Priority bands, highest searched first. Quiet history sits strictly inside
# (-HISTORY_MAX, HISTORY_MAX), below both killer slots and every capture.
TT_MOVE_PRIORITY = 1_000_000
PROMOTION_PRIORITY = 900_000
CAPTURE_PRIORITY = 100_000
KILLER_PRIORITY = (90_000, 80_000)

Killers = tuple[chess.Move | None, chess.Move | None]
NO_KILLERS: Killers = (None, None)


def capture_priority(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: most valuable victim first, then least valuable attacker."""
    victim = board.piece_type_at(move.to_square)
    # An en passant capture lands on an empty square; the victim is a pawn.
    victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
    attacker = board.piece_type_at(move.from_square)
    attacker_value = PIECE_VALUE[attacker] if attacker else 0
    return CAPTURE_PRIORITY + 10 * victim_value - attacker_value


def move_priority(
    board: chess.Board,
    move: chess.Move,
    tt_move: chess.Move | None,
    killers: Killers,
    history: HistoryTable | None,
) -> int:
    """Higher is searched earlier.

    TT move > promotions > captures (MVV-LVA) > killer 0 > killer 1 > quiet by history.
    History only ever decides the order among quiet moves.
    """
    if move == tt_move:
        return TT_MOVE_PRIORITY
    if move.promotion:
        return PROMOTION_PRIORITY + PIECE_VALUE[move.promotion]
    if board.is_capture(move):
        return capture_priority(board, move)
    if move == killers[0]:
        return KILLER_PRIORITY[0]
    if move == killers[1]:
        return KILLER_PRIORITY[1]
    if history is None:
        return 0
    return history.get(board.turn, move)


def ordered_moves(
    board: chess.Board,
    tt_move: chess.Move | None = None,
    killers: Killers = NO_KILLERS,
    history: HistoryTable | None = None,
) -> list[chess.Move]:
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: move_priority(board, move, tt_move, killers, history), reverse=True)
    return moves


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    """Quiet moves are the only ones killers and history are allowed to learn from."""
    return not move.promotion and not board.is_capture(move)


# ---------------------------------------------------------------------------
# 2 + 3 + 7. Negamax with alpha-beta, quiescence, TT and killers
# ---------------------------------------------------------------------------

MAX_PLY = 128


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One searcher per game. The TT and history persist across moves; killers per search.

    Feature flags exist so tests can compare the search with and without each source of
    ordering: the result must be the same, only the node count may change.
    """

    def __init__(
        self,
        *,
        use_history: bool = True,
        use_killers: bool = True,
        use_tt: bool = True,
    ) -> None:
        self.use_history = use_history
        self.use_killers = use_killers
        self.use_tt = use_tt
        self.history = HistoryTable()
        self.tt: dict[int, TTEntry] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.deadline = float("inf")
        self.nodes = 0

    # -- bookkeeping ----------------------------------------------------------

    def new_search(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        if len(self.tt) > TT_MAX_ENTRIES:
            self.tt.clear()

    def _tick(self) -> None:
        # time.monotonic is cheap, but not free; check every 1024 nodes.
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def _killers_at(self, ply: int) -> Killers:
        if not self.use_killers:
            return NO_KILLERS
        slots = self.killers[ply]
        return (slots[0], slots[1])

    def _store_killer(self, ply: int, move: chess.Move) -> None:
        slots = self.killers[ply]
        if slots[0] != move:
            slots[1] = slots[0]
            slots[0] = move

    def _probe(self, key: int) -> TTEntry | None:
        if not self.use_tt:
            return None
        entry = self.tt.get(key)
        return entry if entry is not None and entry.key == key else None

    def _store(
        self, key: int, depth: int, score: int, flag: int, best_move: chess.Move | None, ply: int
    ) -> None:
        if not self.use_tt:
            return
        existing = self.tt.get(key)
        if existing is not None and existing.depth > depth:
            return  # keep the deeper result
        self.tt[key] = TTEntry(key, depth, _to_tt(score, ply), flag, best_move)

    # -- search ---------------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Search only captures and promotions until the position is quiet."""
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta or ply >= MAX_PLY - 1:
            return stand_pat
        alpha = max(alpha, stand_pat)

        best = stand_pat
        for move in ordered_moves(board):
            if is_quiet(board, move):
                break  # ordered_moves puts every capture and promotion first
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

        # Draws are checked before anything else so we never "win" material in a line
        # the opponent can simply repeat out of.
        if ply > 0 and (board.is_repetition(2) or board.halfmove_clock >= 100):
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        original_alpha = alpha
        key = chess.polyglot.zobrist_hash(board) if self.use_tt else 0
        tt_move: chess.Move | None = None
        entry = self._probe(key)
        if entry is not None:
            tt_move = entry.best_move
            if entry.depth >= depth and ply > 0:
                score = _from_tt(entry.score, ply)
                if entry.flag == EXACT:
                    return score
                if entry.flag == LOWERBOUND:
                    alpha = max(alpha, score)
                elif entry.flag == UPPERBOUND:
                    beta = min(beta, score)
                if alpha >= beta:
                    return score

        history = self.history if self.use_history else None
        moves = ordered_moves(board, tt_move, self._killers_at(ply), history)
        if not moves:
            # Checkmated: losing, and sooner is worse, so mates nearer the root are bigger.
            return -(MATE_SCORE - ply) if board.is_check() else 0

        best = -INFINITY
        best_move: chess.Move | None = None
        tried_quiet: list[chess.Move] = []
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                # The opponent will never allow this line. If a quiet move refuted it,
                # remember that: as a killer for this ply, and in the history table.
                if is_quiet(board, move):
                    if self.use_killers:
                        self._store_killer(ply, move)
                    if self.use_history:
                        self.history.update(board.turn, move, tried_quiet, depth)
                break
            if is_quiet(board, move):
                tried_quiet.append(move)

        if best <= original_alpha:
            flag = UPPERBOUND
        elif best >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        self._store(key, depth, best, flag, best_move, ply)
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, int]:
        """Negamax at the root, but remember which move produced the best score."""
        key = chess.polyglot.zobrist_hash(board) if self.use_tt else 0
        entry = self._probe(key)
        tt_move = entry.best_move if entry is not None else None
        history = self.history if self.use_history else None
        moves = ordered_moves(board, tt_move, self._killers_at(0), history)
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

        self._store(key, depth, best_score, EXACT, best_move, 0)
        return best_move, best_score


# ---------------------------------------------------------------------------
# 4. Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 40
MAX_BUDGET_MS = 8_000


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(
    board: chess.Board, time_left_ms: int, searcher: Searcher, verbose: bool = False
) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth."""
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher.new_search(deadline)
    searcher.history.age()  # last move's cutoffs still matter, just less

    # Depth 1 always completes so there is always something to play.
    best_move, best_score = searcher.search_root(board, 1)

    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        if verbose:
            print(
                f"depth {depth:2d}  score {best_score:8d}  best {best_move.uci()}  "
                f"nodes {searcher.nodes}"
            )
        if abs(best_score) >= MATE_SCORE - MAX_PLY:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


# One searcher per process: the process lives for one game, so the TT and history
# tables carry over from move to move but never leak into another game.
SEARCHER = Searcher()


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms, SEARCHER, verbose=True).uci()
