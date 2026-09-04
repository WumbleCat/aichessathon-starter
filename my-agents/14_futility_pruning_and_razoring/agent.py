"""Futility pruning and razoring chess agent.

Implements my-agents-readmes/14_futility_pruning_and_razoring.md on top of a plain
alpha-beta searcher (material + piece-square evaluation, MVV-LVA / killer / history
move ordering, quiescence search, iterative deepening).

The three selective pruning techniques, each independently switchable through
`PruningConfig`:

  Part A  futility pruning          at shallow depth, when `static_eval + margin <= alpha`,
                                    quiet non-checking non-promoting moves are skipped
  Part B  reverse futility pruning  at shallow depth, when `static_eval - margin >= beta`,
                                    the node is cut off without searching any move
  Part C  razoring                  at shallow depth, when `static_eval + margin < alpha`,
                                    the normal search is replaced by quiescence search
  delta   futility in quiescence    a capture whose victim plus a margin cannot lift the
                                    stand-pat score to alpha is skipped

The main-search parts are disabled while in check, at PV nodes (the leftmost path of the
tree), and whenever the window touches mate scores. The default `Searcher` uses `DEFAULT_CONFIG`;
`Searcher(deadline, config=NO_PRUNING)` gives the unpruned baseline for comparisons.

Lives at my-agents/14_futility_pruning_and_razoring/agent.py so the harness can import
it. To submit, copy it to agent.py at the root of the repo.
"""

from __future__ import annotations

import math
import time
from collections.abc import Hashable
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------------------
# Evaluation: material plus piece-square tables, from the mover's point of view
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Diagram order (rank 8 first, White at the bottom); flipped below to a1..h8 order.
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
MATE_BOUND = MATE_SCORE - 1_000  # anything beyond this is a mate score
INFINITY = math.inf


def evaluate(board: chess.Board) -> int:
    """Static score for the side to move. Positive means the mover is better."""
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == board.turn else -value
    return score


def is_mate_score(score: float) -> bool:
    return abs(score) >= MATE_BOUND


# ---------------------------------------------------------------------------
# Pruning configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PruningConfig:
    """Switches and margins for the selective pruning in `Searcher.negamax`.

    Margins are centipawns. Depth-indexed tuples are read as `margins[depth]`, so
    index 0 is unused and the tuple must be at least `*_depth + 1` long.
    """

    # Part A: skip quiet moves when static_eval + futility_margins[depth] <= alpha.
    futility: bool = True
    futility_depth: int = 2
    futility_margins: tuple[int, ...] = (0, 120, 250)

    # Part B: cut off when static_eval - reverse_futility_margin * depth >= beta.
    reverse_futility: bool = True
    reverse_futility_depth: int = 3
    reverse_futility_margin: int = 100

    # Part C: drop into quiescence when static_eval + razor_margins[depth] < alpha.
    razoring: bool = True
    razor_depth: int = 2
    razor_margins: tuple[int, ...] = (0, 250, 450)

    # Futility inside quiescence (delta pruning): skip a capture when even winning the
    # victim outright, plus a margin, would leave us below alpha.
    delta: bool = True
    delta_margin: int = 200

    def __post_init__(self) -> None:
        if self.futility and len(self.futility_margins) <= self.futility_depth:
            raise ValueError("futility_margins needs an entry for every depth up to futility_depth")
        if self.razoring and len(self.razor_margins) <= self.razor_depth:
            raise ValueError("razor_margins needs an entry for every depth up to razor_depth")

    @property
    def max_prune_depth(self) -> int:
        """Deepest node at which any main-search pruning can fire (0 when all are off)."""
        depths = [
            self.futility_depth if self.futility else 0,
            self.reverse_futility_depth if self.reverse_futility else 0,
            self.razor_depth if self.razoring else 0,
        ]
        return max(depths)


DEFAULT_CONFIG = PruningConfig()
NO_PRUNING = PruningConfig(futility=False, reverse_futility=False, razoring=False, delta=False)


def is_quiet(board: chess.Board, move: chess.Move) -> bool:
    """A move futility pruning is allowed to skip: no capture, promotion, or check."""
    if move.promotion is not None or board.is_capture(move):
        return False
    return not board.gives_check(move)


# ---------------------------------------------------------------------------
# Move ordering: promotions, MVV-LVA captures, killers, history
# ---------------------------------------------------------------------------

MAX_PLY = 128


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square)
    victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]  # en passant
    attacker = board.piece_type_at(move.from_square)
    attacker_value = PIECE_VALUE[attacker] if attacker else 0
    return 10 * victim_value - attacker_value


class MoveOrderer:
    def __init__(self) -> None:
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        # history[colour][from][to]
        self.history: list[list[list[int]]] = [
            [[0] * 64 for _ in range(64)] for _ in range(2)
        ]

    def record_cutoff(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        if board.is_capture(move) or move.promotion is not None:
            return
        slot = self.killers[ply]
        if slot[0] != move:
            slot[1] = slot[0]
            slot[0] = move
        self.history[int(board.turn)][move.from_square][move.to_square] += depth * depth

    def priority(self, board: chess.Board, move: chess.Move, ply: int) -> int:
        if move.promotion is not None:
            return 30_000 + PIECE_VALUE[move.promotion]
        if board.is_capture(move):
            return 20_000 + _mvv_lva(board, move)
        if move in self.killers[ply]:
            return 10_000
        return self.history[int(board.turn)][move.from_square][move.to_square]

    def ordered(
        self, board: chess.Board, ply: int, first: chess.Move | None = None
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self.priority(board, move, ply), reverse=True)
        if first is not None and first in moves:
            moves.remove(first)
            moves.insert(0, first)
        return moves


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


@dataclass
class SearchStats:
    nodes: int = 0
    qnodes: int = 0
    futility_pruned: int = 0
    reverse_futility_cutoffs: int = 0
    razor_drops: int = 0
    delta_pruned: int = 0


QS_MAX_PLY = 4  # check evasions in quiescence stop extending past this


class Searcher:
    """One search for one root position: deadline, statistics, and ordering tables."""

    def __init__(
        self,
        deadline: float,
        config: PruningConfig = DEFAULT_CONFIG,
        game_history: frozenset[Hashable] = frozenset(),
    ) -> None:
        self.deadline = deadline
        self.config = config
        self.game_history = game_history  # positions already seen in this game
        self.stats = SearchStats()
        self.orderer = MoveOrderer()

    @property
    def nodes(self) -> int:
        return self.stats.nodes

    def _tick(self) -> None:
        # Pure-Python search runs at a few thousand nodes per second, so check the
        # clock often: 2048 nodes would overshoot a 300 ms budget by a wide margin.
        self.stats.nodes += 1
        if self.stats.nodes & 255 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def _is_draw(self, board: chess.Board) -> bool:
        """Repetition or fifty-move draw, counting positions from earlier in the game.

        The harness only sends a FEN, so `is_repetition` cannot see the game's history;
        `game_history` fills that gap and stops the bot from repeating a won position.
        """
        if board.halfmove_clock >= 100 or board.is_repetition(2):
            return True
        return board._transposition_key() in self.game_history

    # -- quiescence -------------------------------------------------------

    def _tactical_moves(self, board: chess.Board, ply: int) -> list[chess.Move]:
        """Captures and promotions, best first, without generating every legal move."""
        moves = list(board.generate_legal_captures())
        seventh = chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
        pawns = board.pawns & board.occupied_co[board.turn] & seventh
        if pawns:
            moves.extend(board.generate_legal_moves(pawns, ~board.occupied & chess.BB_ALL))
        moves.sort(key=lambda move: self.orderer.priority(board, move, ply), reverse=True)
        return moves

    def quiescence(
        self, board: chess.Board, alpha: float, beta: float, ply: int, qply: int = 0
    ) -> float:
        """Resolve captures (and check evasions) so the leaf evaluation is stable."""
        self._tick()
        self.stats.qnodes += 1

        in_check = board.is_check()
        if in_check and qply < QS_MAX_PLY:
            moves = self.orderer.ordered(board, ply)
            if not moves:
                return -(MATE_SCORE - ply)
            evasion_best = -INFINITY
            for move in moves:
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
                board.pop()
                if score > evasion_best:
                    evasion_best = score
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            break
            return evasion_best

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        best: float = stand_pat
        alpha = max(alpha, stand_pat)
        delta = self.config.delta and not is_mate_score(alpha)

        for move in self._tactical_moves(board, ply):
            if delta and move.promotion is None:
                victim = board.piece_type_at(move.to_square)
                gain = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
                if stand_pat + gain + self.config.delta_margin <= alpha:
                    self.stats.delta_pruned += 1
                    continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- main search ------------------------------------------------------

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int = 0,
        pv_node: bool = True,
    ) -> float:
        """Fail-soft alpha-beta with futility pruning, reverse futility, and razoring."""
        self._tick()

        if ply > 0 and self._is_draw(board):
            return 0.0

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        in_check = board.is_check()
        config = self.config
        near_mate = is_mate_score(alpha) or is_mate_score(beta)
        # Selective pruning is only considered at shallow, quiet, non-PV, non-mate nodes.
        prunable = (
            depth <= config.max_prune_depth
            and not pv_node
            and not in_check
            and not near_mate
            and ply > 0
        )
        static_eval = evaluate(board) if prunable else 0

        # Part B: reverse futility pruning. We are so far above beta that even a
        # pessimistic margin leaves us with a cutoff; do not bother searching.
        if (
            prunable
            and config.reverse_futility
            and depth <= config.reverse_futility_depth
            and static_eval - config.reverse_futility_margin * depth >= beta
        ):
            self.stats.reverse_futility_cutoffs += 1
            return static_eval

        # Part C: razoring. We are so far below alpha that only tactics could save
        # us; let quiescence look for them instead of the full-width search.
        if (
            prunable
            and config.razoring
            and depth <= config.razor_depth
            and static_eval + config.razor_margins[depth] < alpha
        ):
            score = self.quiescence(board, alpha, beta, ply)
            if depth == 1 or score <= alpha:
                self.stats.razor_drops += 1
                return score

        # Part A: futility pruning decision for this node. Quiet moves are skipped
        # in the loop below, because they cannot plausibly raise alpha.
        futile = (
            prunable
            and config.futility
            and depth <= config.futility_depth
            and static_eval + config.futility_margins[depth] <= alpha
        )

        moves = self.orderer.ordered(board, ply)
        if not moves:
            return -(MATE_SCORE - ply) if in_check else 0.0

        best = -INFINITY
        best_move: chess.Move | None = None
        for index, move in enumerate(moves):
            # Always search the first move so the node has a real score to return.
            if futile and index > 0 and is_quiet(board, move):
                self.stats.futility_pruned += 1
                continue

            board.push(move)
            score = -self.negamax(
                board, depth - 1, -beta, -alpha, ply + 1, pv_node and index == 0
            )
            board.pop()

            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        self.orderer.record_cutoff(board, move, depth, ply)
                        break

        assert best_move is not None
        return best

    def search_root(
        self, board: chess.Board, depth: int, first: chess.Move | None = None
    ) -> tuple[chess.Move, float]:
        """Full-window search at the root; returns the best move and its score."""
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY

        for index, move in enumerate(self.orderer.ordered(board, 0, first)):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1, index == 0)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        assert best_move is not None, "search_root called with no legal moves"
        return best_move, best_score


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30
MIN_BUDGET_MS = 50
MAX_BUDGET_MS = 8_000


def move_budget_ms(time_left_ms: int) -> int:
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    config: PruningConfig = DEFAULT_CONFIG,
    game_history: frozenset[Hashable] = frozenset(),
) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth."""
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    # Depth 1 runs unclocked so there is always a move; the clock is armed after it.
    searcher = Searcher(INFINITY, config, game_history)
    best_move, best_score = searcher.search_root(board, 1)
    searcher.deadline = deadline

    root_plies = len(board.move_stack)
    for depth in range(2, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth, first=best_move)
        except OutOfTime:
            # The exception unwinds through pushed moves; put the board back.
            while len(board.move_stack) > root_plies:
                board.pop()
            break
        stats = searcher.stats
        print(
            f"depth {depth:2d}  score {best_score:8.0f}  best {best_move.uci()}  "
            f"nodes {stats.nodes}  fp {stats.futility_pruned}  "
            f"rfp {stats.reverse_futility_cutoffs}  razor {stats.razor_drops}  "
            f"delta {stats.delta_pruned}"
        )
        if is_mate_score(best_score):
            break
    return best_move


class GameMemory:
    """Positions seen so far in the current game, rebuilt from the FENs we are sent.

    The process lives for exactly one game, but the platform never says when a game
    starts, so a position with fewer plies than the last one we saw means a new game.
    """

    def __init__(self) -> None:
        self.seen: set[Hashable] = set()
        self.last_ply = -1

    def record(self, board: chess.Board) -> frozenset[Hashable]:
        if board.ply() <= self.last_ply:
            self.seen.clear()
        self.last_ply = board.ply()
        self.seen.add(board._transposition_key())
        return frozenset(self.seen)

    def record_reply(self, board: chess.Board, move: chess.Move) -> None:
        board.push(move)
        self.seen.add(board._transposition_key())
        self.last_ply = board.ply()
        board.pop()


_memory = GameMemory()


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    history = _memory.record(board)
    move = choose_move(board, time_left_ms, game_history=history)
    _memory.record_reply(board, move)
    return move.uci()
