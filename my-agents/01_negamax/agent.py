"""Stage 01: plain negamax.

Specification: my-agents-readmes/01_negamax.md.

Negamax collapses the maximising and minimising halves of minimax into one recursive
function using the zero-sum identity

    score(position, for me) = -score(position, for opponent)

Every score in this file is relative to the side to move at that node: positive means the
mover is better. A child's score is negated once when it is handed back to the parent, and
that is the only place perspective changes hands.

This bot is deliberately minimal. There is no pruning, no move ordering, no quiescence and
no transposition table; those are later stages. What it does have is the piece every later
stage is built on: a correct, deterministic, fixed-depth search that never corrupts the
board and always answers with a legal move.

Runs under the harness from this directory:

    uv run python -m harness.play --white my-agents/01_negamax --black baselines/greedy
"""

import time

import chess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF = 10**9
MATE_SCORE = 100_000
DRAW_SCORE = 0

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
MOBILITY_WEIGHT = 2

# The search depth is configurable. Plain negamax visits roughly b**depth nodes for a
# branching factor b, and python-chess manages only a few thousand nodes a second, so depth 3
# fits a move's budget only when the position is narrow enough and depth 4 never does.
DEFAULT_DEPTH = 3
MAX_DEPTH = 3

# Time management. A move's budget is a slice of the clock. The depth is the deepest one whose
# estimated node count, b**depth at the root's branching factor b, fits the budget at the node
# cost measured at import. The root loop also stops early once the hard limit passes, so a
# mis-estimate costs strength, not the game.
MOVES_TO_GO = 30
HARD_STOP_FRACTION = 2.0
# Bounds on the measured node cost, so a noisy calibration cannot pick an absurd depth.
MIN_MS_PER_NODE = 0.05
MAX_MS_PER_NODE = 5.0


class SearchStats:
    """Counters for benchmarking: nodes visited, depth searched, elapsed wall time."""

    def __init__(self) -> None:
        self.nodes = 0
        self.depth = 0
        self.elapsed_s = 0.0

    def __repr__(self) -> str:
        nps = self.nodes / self.elapsed_s if self.elapsed_s > 0 else 0.0
        return (
            f"depth {self.depth} nodes {self.nodes} "
            f"time {self.elapsed_s * 1000:.0f}ms ({nps:.0f} nodes/s)"
        )


# ---------------------------------------------------------------------------
# Evaluation, from the side to move's point of view
# ---------------------------------------------------------------------------


def material(board: chess.Board, side: chess.Color) -> int:
    """Material balance in centipawns, positive when `side` has more."""
    own = board.occupied_co[side]
    balance = 0
    for piece, value in PIECE_VALUE.items():
        pieces = board.pieces_mask(piece, chess.WHITE) | board.pieces_mask(piece, chess.BLACK)
        balance += value * ((pieces & own).bit_count() - (pieces & ~own).bit_count())
    return balance


def mobility(board: chess.Board, side: chess.Color) -> int:
    """Squares attacked by `side`'s pieces, other than pawns, that its own men do not occupy.

    This is a cheap stand-in for a legal move count. It needs no move generation, which is
    what makes the leaves affordable: a legal move list costs more than everything else at
    a leaf put together.
    """
    own = board.occupied_co[side]
    movers = own & ~board.pawns
    squares = 0
    for square in chess.scan_forward(movers):
        squares += (board.attacks_mask(square) & ~own).bit_count()
    return squares


def evaluate(board: chess.Board) -> int:
    """Static score, positive when the side to move is better.

    Material plus a small mobility term. Stage 03 replaces this with a real evaluation;
    what matters here is the convention: the score belongs to the side to move.
    """
    mover = board.turn
    return material(board, mover) + MOBILITY_WEIGHT * (
        mobility(board, mover) - mobility(board, not mover)
    )


def is_drawn(board: chess.Board) -> bool:
    """Draws the rules decide without a move: dead material and the fifty-move rule.

    Repetition is not checked here. A search board starts from a FEN with no history, so
    within a shallow search a threefold repetition cannot occur.
    """
    return board.is_insufficient_material() or board.halfmove_clock >= 100


# ---------------------------------------------------------------------------
# Negamax
# ---------------------------------------------------------------------------


def negamax(board: chess.Board, depth: int, ply: int = 0, stats: SearchStats | None = None) -> int:
    """Score `board` for the side to move by searching `depth` plies.

    Terminal positions are scored before the static evaluation regardless of depth, so a
    mate on the horizon is never mistaken for a quiet position. Mate scores are adjusted
    by ply: the mated side scores -MATE_SCORE + ply, so a sooner mate is worse for the
    loser and, once negated at the parent, better for the winner.

    The board is restored before returning: every push has a matching pop.
    """
    if stats is not None:
        stats.nodes += 1

    if depth == 0:
        # A leaf only has to know whether any legal move exists, which python-chess
        # answers without building the whole list.
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or is_drawn(board):
            return DRAW_SCORE
        return evaluate(board)

    moves = list(board.legal_moves)
    if not moves:
        # No legal move: checkmate if in check, stalemate otherwise.
        return -MATE_SCORE + ply if board.is_check() else DRAW_SCORE
    if is_drawn(board):
        return DRAW_SCORE

    best_score = -INF
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, ply + 1, stats)
        board.pop()
        best_score = max(best_score, score)
    return best_score


def position_key(board: chess.Board) -> str:
    """The part of a position that repetition compares: placement, turn, castling, ep."""
    return board.epd()


def search_root(
    board: chess.Board,
    depth: int,
    deadline: float | None = None,
    stats: SearchStats | None = None,
    seen: dict[str, int] | None = None,
) -> tuple[chess.Move | None, int]:
    """Search every legal move to `depth` and return (best move, its score).

    Returns (None, terminal score) when there is no legal move. Ties keep the first move
    in python-chess's generation order, so identical inputs give identical outputs.

    `deadline` is a time.monotonic() value. Once it passes, the moves not yet searched are
    skipped and the best move found so far is returned. The first move is always searched,
    so a move comes back whenever one exists.

    `seen` counts the positions this game has already reached. A root move that returns to
    one of them is scored as a draw, so the engine neither repeats when ahead nor avoids a
    repetition when behind. The search board comes from a FEN without history, so this is
    the only place repetition can be seen at all.
    """
    if stats is not None:
        stats.depth = depth

    best_move: chess.Move | None = None
    best_score = -INF
    for move in board.legal_moves:
        if best_move is not None and deadline is not None and time.monotonic() >= deadline:
            break
        board.push(move)
        if seen and seen.get(position_key(board), 0) > 0:
            score = DRAW_SCORE
        else:
            score = -negamax(board, depth - 1, 1, stats)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move

    if best_move is None:
        return None, negamax(board, 0, 0, stats)
    return best_move, best_score


# ---------------------------------------------------------------------------
# Time management and the harness entrypoint
# ---------------------------------------------------------------------------


def calibrate_ms_per_node() -> float:
    """Measure the cost of one node on this machine with a short search from the start.

    Runs once at import, inside the platform's init budget, so the depth chosen on the
    clock reflects the core the game is actually played on.
    """
    stats = SearchStats()
    started = time.monotonic()
    search_root(chess.Board(), 2, stats=stats)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    measured = elapsed_ms / max(stats.nodes, 1)
    return min(max(measured, MIN_MS_PER_NODE), MAX_MS_PER_NODE)


MS_PER_NODE = calibrate_ms_per_node()

# Positions reached so far in this game, for repetition scoring at the root. The process is
# started fresh for every game, so this never carries over.
GAME_HISTORY: dict[str, int] = {}


def reset_game() -> None:
    """Forget the positions of the current game. Tests use it; the platform never needs to."""
    GAME_HISTORY.clear()


def choose_depth(branching: int, time_left_ms: int, ms_per_node: float = MS_PER_NODE) -> int:
    """Pick the deepest search expected to fit in this move's share of the clock.

    `branching` is the number of legal moves at the root. A depth d search visits about
    branching**d nodes, so d is raised while that estimate still fits the budget.
    """
    budget_ms = time_left_ms / MOVES_TO_GO
    branching = max(branching, 1)
    depth = 1
    for candidate in range(2, MAX_DEPTH + 1):
        if branching**candidate * ms_per_node <= budget_ms:
            depth = candidate
    return depth


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    board = chess.Board(fen)
    key = position_key(board)
    GAME_HISTORY[key] = GAME_HISTORY.get(key, 0) + 1
    depth = choose_depth(board.legal_moves.count(), time_left_ms)
    budget_s = time_left_ms / MOVES_TO_GO / 1000.0
    stats = SearchStats()

    started = time.monotonic()
    move, score = search_root(
        board,
        depth,
        deadline=started + budget_s * HARD_STOP_FRACTION,
        stats=stats,
        seen=GAME_HISTORY,
    )
    stats.elapsed_s = time.monotonic() - started

    if move is None:
        # The harness never asks for a move in a finished game, but never answer with junk.
        raise ValueError(f"no legal move in {fen}")

    board.push(move)
    key = position_key(board)
    GAME_HISTORY[key] = GAME_HISTORY.get(key, 0) + 1
    board.pop()

    # Flushed, so the line survives the runner's pipe when the harness ends the game.
    print(f"01_negamax: {move.uci()} score {score} {stats!r}", flush=True)
    return move.uci()
