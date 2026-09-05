"""AI Chessathon entrypoint for the 20_pvs engine.

get_move(fen, time_left_ms) -> UCI move.

Safety net: a legal fallback move is chosen with python-chess BEFORE the engine runs, the
engine's answer is verified against python-chess's legal moves, and any exception inside
the engine falls back to the python-chess move. The numba JIT is compiled and warmed at
import time (inside the 90 s initialisation budget).
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess  # noqa: E402

_IMPORT_START = time.perf_counter()
_ENGINE_OK = False
_ENGINE_ERROR = ""
try:
    from pvs_board import ST_HASH, Position, move_to_uci  # noqa: E402
    from pvs_search import Searcher  # noqa: E402

    _ENGINE_OK = True
except Exception as exc:  # pragma: no cover - only on a broken platform
    _ENGINE_ERROR = repr(exc)

# ------------------------------------------------------------------ time control

INCREMENT_MS = 500
MIN_THINK_MS = 15
PANIC_MS = 120  # below this we only play the instant fallback or a 1-ply search


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


# --------------------------------------------------------------------- game state

_searcher: Searcher | None = None
_position: Position | None = None
_history_keys: list[int] = []
_last_fen: str | None = None


def _engine_move(board: chess.Board, fen: str, time_left_ms: int) -> str | None:
    global _searcher, _position, _history_keys, _last_fen
    if not _ENGINE_OK:
        return None
    if _searcher is None:
        _searcher = Searcher()
        _position = Position(fen)
        _history_keys = []
    assert _position is not None
    _position.set_fen(fen)
    key = int(_position.st[ST_HASH])
    # a new game in the same process (should not happen on the platform) resets the memory
    if board.fullmove_number <= 1 and board.turn == chess.WHITE and _history_keys:
        _history_keys = []
        _searcher.clear()
    budget_ms = think_time_ms(time_left_ms, board.fullmove_number)
    if budget_ms <= 0:
        return None
    move, score, depth, info = _searcher.search(
        _position,
        time_budget=budget_ms / 1000.0,
        max_depth=64,
        history_keys=_history_keys,
    )
    if move == 0:
        return None
    # remember this position and the one after our move for repetition detection
    _history_keys.append(key)
    _position.push(move)
    _history_keys.append(int(_position.st[ST_HASH]))
    _position.pop()
    _last_fen = fen
    print(
        f"20_pvs depth {depth}/{info.get('seldepth', 0)} score {score} nodes {info.get('nodes', 0)} "
        f"nps {info.get('nps', 0)} time {info.get('time', 0):.2f}s clock {time_left_ms}ms"
    )
    return move_to_uci(move)


def _fallback_move(board: chess.Board) -> chess.Move:
    """Cheap one-ply choice: mate > best capture > castling > anything."""
    best, best_score = None, -1
    for move in board.legal_moves:
        score = 0
        victim = board.piece_type_at(move.to_square)
        if victim:
            score += {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                      chess.QUEEN: 9}[victim] * 10
        if move.promotion == chess.QUEEN:
            score += 80
        if board.is_castling(move):
            score += 5
        if score > best_score:
            best, best_score = move, score
    assert best is not None
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    fallback = _fallback_move(board)
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


def _warm_up() -> None:
    """Compile every jitted path once so the first real move pays nothing."""
    global _searcher, _position
    if not _ENGINE_OK:
        return
    _searcher = Searcher()
    _position = Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    _searcher.search(_position, time_budget=0.3, max_depth=6)
    _position.set_fen("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
    _searcher.search(_position, time_budget=0.1, max_depth=6)
    _searcher.clear()
    _searcher.age = 0


_warm_up()
print(f"20_pvs ready in {time.perf_counter() - _IMPORT_START:.1f}s (engine_ok={_ENGINE_OK} {_ENGINE_ERROR})")
