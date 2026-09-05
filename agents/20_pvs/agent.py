"""AI Chessathon entrypoint for the 20_pvs engine.

get_move(fen, time_left_ms) -> UCI move.

Safety net: a legal fallback move is chosen with python-chess BEFORE the engine runs, the
engine's answer is verified against python-chess's legal moves, and any exception inside
the engine falls back to the python-chess move.

The numba JIT (pvs_board + pvs_search) is compiled and warmed in a background thread that
starts at import. On an idle core this finishes well inside the 90 s initialisation
budget; on an overloaded machine the import returns after NUMBA_WAIT_S regardless, and the
first calls to get_move wait a bounded time for the compile before falling back to the
one-ply python-chess move. The engine can therefore never lose a game by exceeding the
init budget, only play weaker moves until it is ready.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess

_IMPORT_START = time.perf_counter()

# ------------------------------------------------------------------ time control

INCREMENT_MS = 500
MIN_THINK_MS = 15
PANIC_MS = 120  # below this we only play the instant fallback
NUMBA_WAIT_S = 60.0  # how long import blocks on the compile thread (init budget is 90 s)
COMPILE_WAIT_FRACTION = 0.3  # share of the clock a move may spend waiting for the compile
COMPILE_WAIT_MAX_S = 40.0
COMPILE_WAIT_MIN_CLOCK_MS = 30_000  # on a short clock, do not wait: play the fallback at once


def think_time_ms(time_left_ms: int, move_number: int) -> float:
    """Milliseconds to spend on this move (soft budget; the search stops itself)."""
    if time_left_ms <= PANIC_MS:
        return 0.0
    # assume ~35 more moves, plus most of the increment, never more than a quarter of the clock
    remaining_moves = max(20, 40 - move_number // 2)
    budget = time_left_ms / remaining_moves + INCREMENT_MS * 0.8
    budget = min(budget, time_left_ms / 4.0)
    # reserve the watchdog margin: python overhead + last iteration overshoot
    budget = min(budget, time_left_ms - 100)
    return max(MIN_THINK_MS, budget)


# --------------------------------------------------------------------- engine load

_ENGINE_OK = False
_ENGINE_ERROR = ""
_ENGINE_LOAD_S = 0.0
_engine: dict[str, Any] = {}  # Searcher, Position, ST_HASH, move_to_uci once compiled
_searcher: Any = None
_position: Any = None
_history_keys: list[int] = []


def _load_engine() -> None:
    """Thread target: import (= compile) the jitted modules and warm every search path."""
    global _ENGINE_OK, _ENGINE_ERROR, _ENGINE_LOAD_S, _searcher, _position
    started = time.perf_counter()
    try:
        from pvs_board import ST_HASH, Position, move_to_uci
        from pvs_search import Searcher

        searcher = Searcher()
        position = Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        searcher.search(position, time_budget=0.3, max_depth=6)
        position.set_fen("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
        searcher.search(position, time_budget=0.1, max_depth=6)
        searcher.clear()
        searcher.age = 0
        _engine.update(
            ST_HASH=ST_HASH, Position=Position, move_to_uci=move_to_uci, Searcher=Searcher
        )
        _searcher = searcher
        _position = position
        _ENGINE_OK = True
    except Exception as exc:  # pragma: no cover - only on a broken platform
        _ENGINE_ERROR = repr(exc)
    _ENGINE_LOAD_S = time.perf_counter() - started


_load_thread = threading.Thread(target=_load_engine, name="20_pvs-compile", daemon=True)
_load_thread.start()


def _engine_ready(time_left_ms: int) -> bool:
    """True once the compile thread is done; waits a bounded share of the clock for it."""
    if _load_thread.is_alive():
        wait = 0.0
        if time_left_ms >= COMPILE_WAIT_MIN_CLOCK_MS:
            wait = min(COMPILE_WAIT_MAX_S, time_left_ms / 1000.0 * COMPILE_WAIT_FRACTION)
            _load_thread.join(wait)
        if _load_thread.is_alive():
            print(f"20_pvs still compiling after {wait:.0f}s wait, playing fallback")
            return False
    return _ENGINE_OK


# --------------------------------------------------------------------- game state


def _engine_move(board: chess.Board, fen: str, time_left_ms: int) -> str | None:
    global _history_keys
    if not _engine_ready(time_left_ms):
        return None
    searcher, position = _searcher, _position
    st_hash = _engine["ST_HASH"]
    position.set_fen(fen)
    key = int(position.st[st_hash])
    # a new game in the same process (should not happen on the platform) resets the memory
    if board.fullmove_number <= 1 and board.turn == chess.WHITE and _history_keys:
        _history_keys = []
        searcher.clear()
    budget_ms = think_time_ms(time_left_ms, board.fullmove_number)
    if budget_ms <= 0:
        return None
    move, score, depth, info = searcher.search(
        position,
        time_budget=budget_ms / 1000.0,
        max_depth=64,
        history_keys=_history_keys,
    )
    if move == 0:
        return None
    # remember this position and the one after our move for repetition detection
    _history_keys.append(key)
    position.push(move)
    _history_keys.append(int(position.st[st_hash]))
    position.pop()
    print(
        f"20_pvs depth {depth}/{info.get('seldepth', 0)} score {score} "
        f"nodes {info.get('nodes', 0)} nps {info.get('nps', 0)} "
        f"time {info.get('time', 0):.2f}s clock {time_left_ms}ms"
    )
    return str(_engine["move_to_uci"](move))


def _fallback_move(board: chess.Board) -> chess.Move | None:
    """Cheap one-ply choice: mate > best capture > castling > anything. None if no moves."""
    best, best_score = None, -1
    for move in board.legal_moves:
        score = 0
        victim = board.piece_type_at(move.to_square)
        if victim:
            score += {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                      chess.QUEEN: 9, chess.KING: 0}[victim] * 10
        if move.promotion == chess.QUEEN:
            score += 80
        if board.is_castling(move):
            score += 5
        if score > best_score:
            best, best_score = move, score
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    fallback = _fallback_move(board)
    if fallback is None:
        return "0000"  # no legal move: the game is already over, nothing sensible to return
    try:
        uci = _engine_move(board, fen, time_left_ms)
    except Exception as exc:  # never lose on a crash
        print(f"20_pvs engine error: {exc!r}")
        uci = None
    if uci is not None:
        try:
            if chess.Move.from_uci(uci) in board.legal_moves:
                return uci
        except ValueError:
            pass
        print(f"20_pvs produced illegal move {uci} in {fen}")
    return fallback.uci()


# --------------------------------------------------------------------- warm-up

_load_thread.join(NUMBA_WAIT_S)
print(
    f"20_pvs import returned in {time.perf_counter() - _IMPORT_START:.1f}s "
    f"(engine_ok={_ENGINE_OK} compiling={_load_thread.is_alive()} {_ENGINE_ERROR})"
)
