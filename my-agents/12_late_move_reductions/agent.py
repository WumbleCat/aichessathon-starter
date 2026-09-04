"""Late Move Reductions (LMR) chess agent.

Stage 12 of the roadmap in my-agents-readmes. The engine underneath is the classical stack
from stages 1 to 11, kept as lean as python-chess allows:

    evaluation            material plus piece-square tables, from the mover's view
    negamax + alpha-beta  one maximising function, child scores negated
    quiescence            captures and promotions resolved at the leaves
    iterative deepening   depth 1, 2, 3 ... until the move budget runs out
    move ordering         TT move, MVV-LVA captures, killers, history
    transposition table   Zobrist keys from python-chess, bounds and best move cached
    PVS                   first move full window, the rest with a null window
    null-move pruning     pass the turn; if still >= beta, prune

Late Move Reductions is the new part. Move ordering puts the likely-best moves first, so a
move that shows up fourth or later in the list is rarely best. Those late quiet moves are
searched one or two plies shallower than the rest. Whenever a reduced search unexpectedly
beats alpha, the move is re-searched at full depth so a strong move is never lost to the
reduction. The whole decision lives in `LmrPolicy`, which is the only thing this stage adds
and is switched off with a single flag for comparison.

Lives at my-agents/12_late_move_reductions/agent.py so the harness can import it. To submit,
copy it to agent.py at the root of the repo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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
# They are flipped below so index 0 is a1, matching python-chess.
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


def _mirror(table: list[int]) -> list[int]:
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_MIDDLE: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING_MIDDLE),
}
_WHITE_END = dict(_WHITE_MIDDLE)
_WHITE_END[chess.KING] = _from_diagram(_KING_END)


def _with_material(tables: dict[chess.PieceType, list[int]]) -> list[list[int]]:
    """Index by piece type (1..6); entry 0 is unused padding."""
    merged: list[list[int]] = [[0] * 64]
    for piece in chess.PIECE_TYPES:
        value = PIECE_VALUE[piece]
        merged.append([value + bonus for bonus in tables[piece]])
    return merged


def _for_black(white_tables: list[list[int]]) -> list[list[int]]:
    return [[0] * 64] + [_mirror(row) for row in white_tables[1:]]


# TABLE[phase][colour][piece][square] -> material plus positional bonus.
# phase 0 middlegame, 1 endgame; colour index matches chess.WHITE == 1.
_MIDDLE_WHITE = _with_material(_WHITE_MIDDLE)
_END_WHITE = _with_material(_WHITE_END)
_TABLES: list[list[list[list[int]]]] = [
    [_for_black(_MIDDLE_WHITE), _MIDDLE_WHITE],
    [_for_black(_END_WHITE), _END_WHITE],
]

MATE_SCORE = 100_000  # bigger than any material swing
MAX_PLY = 128
MATE_BOUND = MATE_SCORE - MAX_PLY  # anything beyond this is a mate score
INFINITY = 10**9
ENDGAME_MATERIAL = 2_600  # total non-pawn material at or below which the king comes out


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move (positive = mover better)."""
    occupied = board.occupied
    white = board.occupied_co[chess.WHITE]
    phase_material = (
        320 * chess.popcount(board.knights)
        + 330 * chess.popcount(board.bishops)
        + 500 * chess.popcount(board.rooks)
        + 900 * chess.popcount(board.queens)
    )
    phase = 1 if phase_material <= ENDGAME_MATERIAL else 0
    tables = _TABLES[phase]
    white_table = tables[1]
    black_table = tables[0]
    piece_type_at = board.piece_type_at
    score = 0
    for square in chess.scan_forward(occupied):
        piece = piece_type_at(square)
        if piece is None:  # pragma: no cover - occupied squares always hold a piece
            continue
        if (white >> square) & 1:
            score += white_table[piece][square]
        else:
            score -= black_table[piece][square]
    return score if board.turn == chess.WHITE else -score


# ---------------------------------------------------------------------------
# Configuration: every selective feature can be switched independently
# ---------------------------------------------------------------------------


@dataclass
class LmrPolicy:
    """Decides which moves are reduced and by how much. Stage 12 lives entirely here.

    The conservative defaults follow the readme: quiet moves at index >= 4 when the
    remaining depth is >= 3 lose one ply. Deep in the tree, very late moves lose a
    second ply. `max_reduction` and the `depth - 2` clamp keep the reduced search from
    dropping straight into quiescence, so a reduced move still gets one real ply.
    """

    enabled: bool = True
    min_depth: int = 3  # only reduce when at least this much depth remains
    full_depth_moves: int = 4  # moves with index < this are always full depth
    base_reduction: int = 1
    extra_late_moves: int = 12  # from this index on, shave another ply ...
    extra_min_depth: int = 6  # ... but only when the search is this deep
    max_reduction: int = 2

    def should_reduce(
        self,
        *,
        move_index: int,
        depth: int,
        is_quiet: bool,
        in_check: bool,
        gives_check: bool,
        is_tt_move: bool,
        is_killer: bool,
        near_mate: bool,
    ) -> bool:
        """The readme's "Do Not Reduce" list, as one predicate."""
        return (
            self.enabled
            and depth >= self.min_depth
            and move_index >= self.full_depth_moves
            and is_quiet  # captures and promotions keep full depth
            and not in_check  # never reduce when we must answer a check
            and not gives_check  # checking moves are forcing; keep them honest
            and not is_tt_move
            and not is_killer
            and not near_mate
        )

    def reduction(self, move_index: int, depth: int) -> int:
        """How many plies to shave. Never leaves less than one ply of search."""
        plies = self.base_reduction
        if move_index >= self.extra_late_moves and depth >= self.extra_min_depth:
            plies += 1
        plies = min(plies, self.max_reduction)
        return max(0, min(plies, depth - 2))


@dataclass
class SearchConfig:
    use_tt: bool = True
    use_killers: bool = True
    use_history: bool = True
    use_pvs: bool = True
    use_null_move: bool = True
    null_move_reduction: int = 2
    null_move_min_depth: int = 3
    lmr: LmrPolicy = field(default_factory=LmrPolicy)


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
TT_MAX_ENTRIES = 1_000_000  # a few hundred MB of Python objects; well under the 2 GB cap

# key -> (depth, score, flag, best_move). Module state survives between moves of one game,
# so lines searched on the previous move are still warm on this one.
TRANSPOSITION_TABLE: dict[int, tuple[int, int, int, chess.Move | None]] = {}


def _score_to_tt(score: int, ply: int) -> int:
    """Mate scores are stored as distance from the *stored* node, not from the root."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
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


@dataclass
class SearchStats:
    nodes: int = 0
    reductions: int = 0  # moves searched at reduced depth
    researches: int = 0  # reduced moves that beat alpha and were searched again
    null_cutoffs: int = 0
    tt_hits: int = 0


_TT_MOVE_BONUS = 1_000_000
_PROMOTION_BONUS = 500_000
_CAPTURE_BONUS = 100_000
_KILLER_BONUS = (90_000, 80_000)
_HISTORY_CAP = 70_000


class Searcher:
    """One search for one root position. Holds the deadline, the heuristics and counters."""

    def __init__(self, deadline: float, config: SearchConfig | None = None) -> None:
        self.deadline = deadline
        self.config = config or SearchConfig()
        self.stats = SearchStats()
        self.tt = TRANSPOSITION_TABLE if self.config.use_tt else {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
        # history[colour][from][to]
        self.history: list[list[list[int]]] = [
            [[0] * 64 for _ in range(64)] for _ in range(2)
        ]
        self.root_move: chess.Move | None = None
        self.check_every = 256  # python-chess is slow, so look at the clock often

    # -- bookkeeping ---------------------------------------------------------

    def _tick(self) -> None:
        self.stats.nodes += 1
        if self.stats.nodes % self.check_every == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def _record_cutoff(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        """A quiet move refuted this node: remember it as a killer and bump its history."""
        if self.config.use_killers:
            slots = self.killers[ply]
            if slots[0] != move:
                slots[1] = slots[0]
                slots[0] = move
        if self.config.use_history:
            row = self.history[int(board.turn)][move.from_square]
            row[move.to_square] = min(_HISTORY_CAP, row[move.to_square] + depth * depth)

    # -- move ordering -------------------------------------------------------

    def _ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[tuple[int, chess.Move]]:
        """(priority, move) pairs, highest first: TT move, promotions, captures, killers,
        then quiet moves by history."""
        killers = self.killers[ply]
        history = self.history[int(board.turn)]
        piece_type_at = board.piece_type_at
        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            if move == tt_move:
                priority = _TT_MOVE_BONUS
            elif move.promotion:
                priority = _PROMOTION_BONUS + PIECE_VALUE[move.promotion]
            else:
                victim = piece_type_at(move.to_square)
                if victim is not None or board.is_en_passant(move):
                    victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
                    attacker = piece_type_at(move.from_square)
                    attacker_value = PIECE_VALUE[attacker] if attacker else 0
                    priority = _CAPTURE_BONUS + 10 * victim_value - attacker_value
                elif move == killers[0]:
                    priority = _KILLER_BONUS[0]
                elif move == killers[1]:
                    priority = _KILLER_BONUS[1]
                else:
                    priority = history[move.from_square][move.to_square]
            scored.append((priority, move))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    # -- quiescence ----------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Resolve captures and promotions so the leaf evaluation is not mid-exchange."""
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta or ply >= MAX_PLY:
            return stand_pat
        alpha = max(alpha, stand_pat)

        piece_type_at = board.piece_type_at
        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            victim = piece_type_at(move.to_square)
            if move.promotion:
                scored.append((_PROMOTION_BONUS + PIECE_VALUE[move.promotion], move))
            elif victim is not None or board.is_en_passant(move):
                victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
                # Delta pruning: even winning this piece outright cannot reach alpha.
                if stand_pat + victim_value + 200 < alpha:
                    continue
                attacker = piece_type_at(move.from_square)
                attacker_value = PIECE_VALUE[attacker] if attacker else 0
                scored.append((10 * victim_value - attacker_value, move))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        for _, move in scored:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    # -- main search ---------------------------------------------------------

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int = 0,
        allow_null: bool = True,
    ) -> int:
        """Best score the side to move can force within `depth` plies.

        Negamax with alpha-beta, PVS, TT, null-move pruning and LMR. At ply 0 the best
        move is written to `self.root_move` so the caller can play it.
        """
        self._tick()
        config = self.config
        is_root = ply == 0

        if not is_root and (board.is_repetition(2) or board.halfmove_clock >= 100):
            return 0
        if ply >= MAX_PLY:
            return evaluate(board)

        in_check = board.is_check()
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        # -- transposition table probe --
        key = chess.polyglot.zobrist_hash(board)
        tt_move: chess.Move | None = None
        entry = self.tt.get(key) if config.use_tt else None
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and not is_root:
                tt_score = _score_from_tt(tt_score, ply)
                self.stats.tt_hits += 1
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        # -- null-move pruning --
        if (
            config.use_null_move
            and allow_null
            and not is_root
            and not in_check
            and not _is_mate_score(beta)
            and depth >= config.null_move_min_depth
            and _has_non_pawn_material(board, board.turn)
        ):
            board.push(chess.Move.null())
            null_depth = depth - 1 - config.null_move_reduction
            score = -self.negamax(board, null_depth, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if score >= beta:
                self.stats.null_cutoffs += 1
                return beta

        moves = self._ordered_moves(board, tt_move, ply)
        if not moves:
            # Checkmated: losing, and sooner is worse, so a mate found nearer the root is
            # a bigger number. Stalemate: draw.
            return -(MATE_SCORE - ply) if in_check else 0

        lmr = config.lmr
        killers = self.killers[ply]
        original_alpha = alpha
        best_score = -INFINITY
        best_move: chess.Move | None = None
        quiets_tried: list[chess.Move] = []

        for move_index, (_, move) in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and not move.promotion

            board.push(move)
            gives_check = board.is_check()
            new_depth = depth - 1

            # -- late move reductions --
            reduction = 0
            if lmr.should_reduce(
                move_index=move_index,
                depth=depth,
                is_quiet=is_quiet,
                in_check=in_check,
                gives_check=gives_check,
                is_tt_move=move == tt_move,
                is_killer=move == killers[0] or move == killers[1],
                near_mate=_is_mate_score(alpha) or _is_mate_score(beta),
            ):
                reduction = lmr.reduction(move_index, depth)

            if move_index == 0 or not config.use_pvs:
                score = -self.negamax(board, new_depth - reduction, -beta, -alpha, ply + 1)
                if reduction > 0:
                    self.stats.reductions += 1
                    if score > alpha:
                        self.stats.researches += 1
                        score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1)
            else:
                # PVS: null window first; a reduced move that beats alpha is re-searched at
                # full depth, and any move that then lands inside the window gets the full
                # window as well.
                score = -self.negamax(board, new_depth - reduction, -alpha - 1, -alpha, ply + 1)
                if reduction > 0:
                    self.stats.reductions += 1
                    if score > alpha:
                        self.stats.researches += 1
                        score = -self.negamax(board, new_depth, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                if is_root:
                    self.root_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if is_quiet:
                        self._record_cutoff(board, move, depth, ply)
                        if config.use_history:
                            # Quiet moves that failed to cut off ahead of this one are
                            # gently penalised so the good one climbs past them next time.
                            row = self.history[int(board.turn)]
                            for quiet in quiets_tried:
                                cell = row[quiet.from_square]
                                cell[quiet.to_square] = max(0, cell[quiet.to_square] - depth)
                    break
            if is_quiet:
                quiets_tried.append(move)

        # -- transposition table store --
        if config.use_tt:
            if best_score <= original_alpha:
                flag = TT_UPPER
            elif best_score >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            if len(self.tt) >= TT_MAX_ENTRIES:
                self.tt.clear()
            old = self.tt.get(key)
            if old is None or old[0] <= depth:
                self.tt[key] = (depth, _score_to_tt(best_score, ply), flag, best_move)
        return best_score

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, int]:
        """Full-window search from the root; returns the best move and its score."""
        self.root_move = None
        score = self.negamax(board, depth, -INFINITY, INFINITY, 0)
        assert self.root_move is not None, "search_root called with no legal moves"
        return self.root_move, score


def _is_mate_score(score: int) -> bool:
    """True for a real mate score; the open-window infinities do not count."""
    return MATE_BOUND <= abs(score) < INFINITY


def _has_non_pawn_material(board: chess.Board, colour: chess.Color) -> bool:
    """Null move is unsafe in pawn endings (zugzwang), so require a piece."""
    pieces = board.occupied_co[colour] & ~board.pawns & ~board.kings
    return pieces != 0


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 30
MAX_BUDGET_MS = 8_000
SAFETY_MS = 60  # keep clear of the referee's clock; process overhead is not free


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    budget = max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))
    return max(1, min(budget, time_left_ms - SAFETY_MS))


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    config: SearchConfig | None = None,
    max_depth: int = MAX_DEPTH,
    verbose: bool = True,
) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth."""
    started = time.monotonic()
    budget_s = move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(started + budget_s, config)

    # Depth 1 is searched without a deadline so there is always something to play.
    searcher.deadline = float("inf")
    best_move, best_score = searcher.search_root(board, 1)
    searcher.deadline = started + budget_s

    for depth in range(2, max_depth + 1):
        # A new iteration usually costs several times the previous one; do not start one
        # that cannot finish.
        elapsed = time.monotonic() - started
        if elapsed > budget_s * 0.4:
            break
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        if verbose:
            stats = searcher.stats
            print(
                f"depth {depth:2d}  score {best_score:7d}  best {best_move.uci()}  "
                f"nodes {stats.nodes}  lmr {stats.reductions}/{stats.researches}  "
                f"null {stats.null_cutoffs}  {elapsed * 1000:.0f}ms"
            )
        if abs(best_score) >= MATE_BOUND:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
