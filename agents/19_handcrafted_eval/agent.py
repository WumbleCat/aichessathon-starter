"""AI Chessathon agent 19: handcrafted tapered evaluation inside an alpha-beta search.

`get_move(fen, time_left_ms)` returns one legal UCI move. The evaluation lives in
`hce_eval.py` (numba-compiled, pure-Python fallback), the search in `hce_search.py`.
This file owns initialisation, time management, the game-history memory used for repetition
detection and the guarantee that something legal is returned no matter what happens inside
the search.

Initialisation: the numba compile of the evaluation normally takes a few seconds, but the
platform gives 90 s and a slow or shared CPU could blow that. The compile therefore runs in a
background thread with a deadline. If it is not finished in time the agent reports ready
anyway and plays with the simple material + piece-square evaluation until the compiled one
is available; every call to `get_move` checks again.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

import chess
import hce_eval_simple
from hce_search import Searcher

# initialisation ----------------------------------------------------------------------------------
# how long import may block waiting for the compiled evaluation (env override for tests)
COMPILE_WAIT_S = float(os.environ.get("HCE_COMPILE_WAIT_S", "55"))

_fast_evaluate: Callable[[chess.Board], int] | None = None
_compile_error: BaseException | None = None
_compiled = threading.Event()


def _build_fast_evaluation() -> None:
    """Import (and so compile) the fast evaluation; on failure retry once without the on-disk
    numba cache, since a broken or unwritable cache is the one failure that is recoverable."""
    global _fast_evaluate, _compile_error
    try:
        for attempt in range(2):
            try:
                import hce_eval

                _fast_evaluate = hce_eval.evaluate_stm
                _compile_error = None
                return
            except BaseException as error:  # any failure means "keep the fallback"
                _compile_error = error
                sys.modules.pop("hce_eval", None)
                if attempt == 0:
                    print(f"compiled evaluation failed ({error!r}); retrying uncached", flush=True)
                    os.environ["HCE_NO_CACHE"] = "1"
    finally:
        _compiled.set()


_compile_started = time.monotonic()
threading.Thread(target=_build_fast_evaluation, name="hce-compile", daemon=True).start()
_compiled.wait(COMPILE_WAIT_S)

# time management -------------------------------------------------------------------------------
# The referee measures wall time and the watchdog grace is only 500 ms, so the budget is derived
# from the clock we were handed and the search checks the clock every 32 nodes.
OVERHEAD_MS = 25  # process/protocol latency we cannot search through
MOVES_TO_GO = 28  # the soft budget is roughly time_left / MOVES_TO_GO
HARD_FACTOR = 3.0  # the hard deadline is this many soft budgets, capped by the clock

_searcher = Searcher(hce_eval_simple.evaluate_stm)
_history: list[tuple[int, object]] = []  # (ply, transposition key) of every position we saw
_INFO = bool(os.environ.get("HCE_INFO"))  # print per-move search statistics to stderr


def _use_best_evaluation() -> str:
    """Swap the compiled evaluation in as soon as it exists. Returns which one is active."""
    if _fast_evaluate is not None and _searcher.evaluate is not _fast_evaluate:
        _searcher.evaluate = _fast_evaluate
        _searcher.tt.clear()  # scores from the fallback evaluation would mislead the search
    return "compiled" if _searcher.evaluate is _fast_evaluate else "simple"


def new_game() -> None:
    _history.clear()
    _searcher.game_keys.clear()


def _remember(board: chess.Board) -> None:
    """Keep the positions of this game so the search can score repetitions as draws."""
    ply = board.ply()
    # a position earlier in the game than the last one we saw means a new game (or a test)
    if _history and ply <= _history[-1][0] - 1:
        new_game()
    if not _history or _history[-1][0] < ply:
        key = board._transposition_key()
        _history.append((ply, key))
        _searcher.game_keys.add(key)


def budget_ms(time_left_ms: int) -> tuple[float, float]:
    """(soft, hard) budgets in milliseconds for this move."""
    usable = max(0.0, time_left_ms - OVERHEAD_MS)
    if usable < 200:
        return usable * 0.15, usable * 0.4
    soft = usable / MOVES_TO_GO
    hard = min(usable * 0.25, soft * HARD_FACTOR)
    return soft, hard


def get_move(fen: str, time_left_ms: int) -> str:
    started = time.monotonic()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:  # the referee never asks in a finished game, but never return nothing
        return "0000"
    fallback = legal[0]
    if len(legal) == 1:
        return fallback.uci()

    try:
        which = _use_best_evaluation()
        _remember(board)
        soft, hard = budget_ms(time_left_ms)
        # the position itself must not count as a repetition of itself during the search
        _searcher.game_keys.discard(board._transposition_key())
        move, score = _searcher.search_root(
            board, started + soft / 1000.0, started + hard / 1000.0
        )
        _searcher.game_keys.add(board._transposition_key())
        if _INFO:
            s = _searcher
            print(
                f"info depth {s.depth_reached} seldepth {s.seldepth} nodes {s.nodes} "
                f"qnodes {s.qnodes} evals {s.eval_calls} tt_hits {s.tt_hits} "
                f"time {(time.monotonic() - started) * 1000:.0f}ms score {score} "
                f"move {move.uci()} eval {which}",
                flush=True,
            )
        if move not in board.legal_moves:
            move = fallback
        board.push(move)
        _remember(board)
        return move.uci()
    except Exception as error:  # a crash loses the game, a fallback move does not
        print(f"search failed: {error!r}", flush=True)
        return fallback.uci()


if _INFO:
    print(
        f"init: evaluation {'compiled' if _fast_evaluate else 'simple'} after "
        f"{time.monotonic() - _compile_started:.1f}s"
        + (f" (compile error: {_compile_error!r})" if _compile_error else ""),
        flush=True,
    )
