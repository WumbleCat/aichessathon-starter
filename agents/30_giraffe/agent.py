"""AI Chessathon entry point: Giraffe-style learned evaluation inside an alpha-beta search.

The platform imports this file once per game and calls ``get_move`` for every move. The
network weights (trained by this project, see ``training/``) are loaded and every numba
function is compiled at import, inside the 90 second init budget. ``giraffe_eval`` holds the
feature extractor and network, ``giraffe_search`` the search. Set ``GIRAFFE_EVAL=hce`` to
run the identical search with the handcrafted control evaluation instead.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import giraffe_eval
except Exception as cache_error:  # noqa: BLE001 - a broken numba cache must not lose the game
    print(f"giraffe: import with numba cache failed ({cache_error!r}); retrying without cache")
    os.environ["GIRAFFE_NUMBA_CACHE"] = "0"
    sys.modules.pop("giraffe_eval", None)
    import giraffe_eval
from giraffe_search import Searcher  # noqa: E402

MODEL_PATH = HERE / "models" / "giraffe.npz"

MAX_DEPTH = 64
MOVES_TO_GO = 30
INCREMENT_SHARE_S = 0.35  # the part of the 0.5 s increment spent every move
MAX_BUDGET_S = 6.0
MIN_BUDGET_S = 0.015
OVERHEAD_MS = 60  # protocol round trip, FEN parsing, clock-check granularity
PANIC_MS = 400  # below this only the unclocked depth-1 search runs
NO_SEARCH_MS = 120  # below this the fallback move goes out untouched


def _load_evaluator() -> tuple[giraffe_eval.Evaluator, str]:
    if os.environ.get("GIRAFFE_EVAL", "net").lower() == "hce" or not MODEL_PATH.exists():
        return giraffe_eval.hce_eval, "hce"
    weights = np.load(MODEL_PATH)["weights"].astype(np.float32)
    return giraffe_eval.NetEvaluator(weights), "net"


_started = time.monotonic()
giraffe_eval.warm_up()
EVALUATE, EVAL_NAME = _load_evaluator()
searcher = Searcher(EVALUATE)
searcher.search(chess.Board(), 0.05, 2)  # warm the search paths too
searcher = Searcher(EVALUATE)
print(f"giraffe: evaluator={EVAL_NAME} init {time.monotonic() - _started:.1f}s")


def budget_seconds(time_left_ms: int) -> float:
    """Wall-clock seconds to spend on this move."""
    usable_ms = max(time_left_ms - OVERHEAD_MS, 0)
    per_move = usable_ms / MOVES_TO_GO + INCREMENT_SHARE_S * 1000.0
    per_move = min(per_move, usable_ms / 4.0)  # never more than a quarter of what is left
    return max(MIN_BUDGET_S, min(MAX_BUDGET_S, per_move / 1000.0))


def fallback_move(board: chess.Board) -> chess.Move:
    """A legal move chosen without searching: best capture or promotion, else the first."""
    best: chess.Move | None = None
    best_value = -1
    for move in board.legal_moves:
        victim = board.piece_type_at(move.to_square)
        value = 0 if victim is None else int(victim)
        if move.promotion == chess.QUEEN:
            value += 10
        if value > best_value:
            best, best_value = move, value
    if best is None:
        raise ValueError("position has no legal moves")
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    move = fallback_move(board)
    searcher.remember(board)
    try:
        if time_left_ms <= NO_SEARCH_MS:
            pass
        elif time_left_ms <= PANIC_MS:
            move, _ = searcher.search(board, 0.0, 1)
        else:
            move, _ = searcher.search(board, budget_seconds(time_left_ms), MAX_DEPTH)
    except Exception as error:  # noqa: BLE001 - a crash loses the game, a fallback does not
        print(f"giraffe: search failed ({error!r}), playing fallback")
    board = chess.Board(fen)  # the search unwinds its moves, but never trust that blindly
    if move not in board.legal_moves:
        move = fallback_move(board)
    board.push(move)
    searcher.remember(board)
    return move.uci()
