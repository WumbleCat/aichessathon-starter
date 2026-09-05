"""27_chessformer submission entrypoint.

A Chessformer-style transformer (trained by this project from random initialisation, see
training/) provides move priors to an alpha-beta search. If no model file ships, the search runs
on its handcrafted evaluation alone.
"""

import os
import sys
import time

# one core: never let BLAS or torch spin up worker threads (must precede the numpy import)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import chess  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cf_search import Searcher  # noqa: E402

MODEL_PATH = os.path.join(_HERE, "models", "chessformer.onnx")
POLICY_MIN_DEPTH = int(os.environ.get("CF_POLICY_MIN_DEPTH", "3"))
USE_MODEL = os.environ.get("CF_USE_MODEL", "1") == "1"

_policy = None
if USE_MODEL and os.path.exists(MODEL_PATH):
    try:
        from cf_infer import PolicyModel

        _policy = PolicyModel(MODEL_PATH)
        _policy.warm_up()
    except Exception as exc:  # the search still works without the model
        print(f"model unavailable: {exc!r}")
        _policy = None

_searcher = Searcher(
    policy_fn=_policy.priors if _policy is not None else None,
    policy_min_depth=POLICY_MIN_DEPTH,
)
_move_count = 0

# safety margin for process/protocol overhead and the coarse deadline checks
SAFETY_MS = 60


def _budget_ms(time_left_ms: int, board: chess.Board) -> float:
    """Milliseconds to spend on this move (soft budget; the search stops well before the clock).

    The contest clock is 120 s + 0.5 s per move, so once the clock is low the increment alone
    sustains roughly 400 ms per move; the fast local arena (10 s + 0.1 s) is handled by the
    proportional term.
    """
    if time_left_ms <= 150:
        return 5.0
    pieces = chess.popcount(board.occupied)
    moves_left = 28 if pieces > 20 else 22
    base = time_left_ms / moves_left
    if time_left_ms < 8000:
        base = max(base, time_left_ms / 10.0)
    budget = min(base, time_left_ms * 0.25) - SAFETY_MS
    return max(5.0, budget)


def get_move(fen: str, time_left_ms: int) -> str:
    global _move_count
    t0 = time.perf_counter()
    board = chess.Board(fen)
    _move_count += 1
    _searcher.note_position(board)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    fallback = legal[0].uci()
    try:
        budget = _budget_ms(time_left_ms, board) / 1000.0
        result = _searcher.search(board, budget)
        move = result.move if result.move is not None else legal[0]
        if move not in board.legal_moves:
            move = legal[0]
        board.push(move)
        _searcher.note_position(board)
        board.pop()
        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"move {_move_count} {move.uci()} depth {result.depth}/{result.seldepth} score "
            f"{result.score} nodes {result.nodes} q {result.qnodes} nps {result.nps:.0f} "
            f"policy_calls {_searcher.policy_calls} budget {budget * 1000:.0f}ms took {elapsed:.0f}ms"
        )
        return move.uci()
    except Exception as exc:
        print(f"search failed: {exc!r}")
        return fallback


# warm the code paths once at import so the first move does not pay for them
_searcher.search(chess.Board(), 0.3, max_depth=3)
_searcher.tt.clear()
_searcher.game_history.clear()
