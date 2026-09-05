"""AI Chessathon entry point: self-trained NNUE evaluation inside an own PVS search.

    get_move(fen, time_left_ms) -> uci

Pipeline per call: python-chess parses the FEN and gives us a guaranteed-legal fallback move,
the position is copied into our numba bitboard engine (cboard), the search (csearch) runs
with the NNUE evaluator (nnue) until its time budget is spent, and the chosen move is
validated against python-chess before it is returned.  Nothing here reads the network or
calls an external engine; the weights in ``weights/`` were trained by this project
(see training/ and models/PROVENANCE.md).

Start-up: numba compiles the engine on first use, which can take longer than the platform's
90 s import budget on a slow core.  The compile therefore runs in a background thread while
import waits for it up to ``INIT_WAIT_S``.  If it is still running when the first moves are
requested, a small pure-python alpha-beta (``_python_search``) answers them; the compiled
engine takes over as soon as it is ready.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

_INIT_START = time.perf_counter()
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# numba may cache compiled code (opt-in, see jitconf); keep it in a writable temp dir
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "nnue21_numba_cache"))

import chess  # noqa: E402

import cboard  # noqa: E402
import csearch  # noqa: E402
import nnue  # noqa: E402

INIT_WAIT_S = 70.0  # wall seconds after import start that we wait for the compile
MIN_BUDGET_S = 0.004
OVERHEAD_S = 0.015  # python-chess parsing, validation, protocol
MATE = 30000

WEIGHTS = nnue.default_weights_path()
NET = nnue.load_net(WEIGHTS) if os.path.exists(WEIGHTS) else None
SEARCHER = csearch.Searcher(NET, use_nnue=NET is not None)

# Positions this process has been asked about (and the ones our replies produced), so the
# search can see game-level repetitions.  The harness only sends FENs.  FENs are kept until the
# compiled engine can turn them into its own zobrist keys.
_HISTORY_FENS: list[str] = []
_HISTORY_KEYS: list[int] = []
_LAST_FEN: str | None = None
_LAST_REPLY: chess.Move | None = None
_STATS: dict[str, float] = {
    "moves": 0,
    "nodes": 0,
    "time": 0.0,
    "max_time": 0.0,
    "depth_sum": 0,
    "fallback": 0,
}

_ENGINE_READY = threading.Event()
_ENGINE_ERROR: BaseException | None = None


def budget_seconds(time_left_ms: int) -> float:
    """Seconds the search may use.  Assumes 0.5 s increments and about 30 more moves."""
    t = time_left_ms / 1000.0
    if t <= 0.15:
        return 0.0  # node-limited emergency search
    if t < 2.0:
        return max(MIN_BUDGET_S, t * 0.04)
    b = t / 30.0 + 0.25
    return min(b, t * 0.2) - OVERHEAD_S


# ----------------------------------------------------------------------------- fallback

_PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _fallback_move(board: chess.Board) -> chess.Move | None:
    """A legal move that exists before any expensive work: best capture by victim value.
    None only when the position has no legal move at all (the game is already over)."""
    best: chess.Move | None = None
    best_value = -1
    for move in board.legal_moves:
        value = 0
        victim = board.piece_type_at(move.to_square)
        if victim is not None:
            value = _PIECE_CP[victim]
        if move.promotion == chess.QUEEN:
            value += 800
        if value > best_value:
            best_value = value
            best = move
    return best


def _material(board: chess.Board) -> int:
    """Material from the side to move's point of view, centipawns."""
    s = 0
    for piece_type, cp in _PIECE_CP.items():
        s += cp * (
            len(board.pieces(piece_type, board.turn))
            - len(board.pieces(piece_type, not board.turn))
        )
    return s


class _Deadline(Exception):
    pass


def _python_search(board: chess.Board, deadline: float) -> chess.Move:
    """Pure python-chess iterative-deepening alpha-beta on material, used only while the
    compiled engine is still being built.  Always returns a legal move."""
    depth0 = len(board.move_stack)
    ordered = sorted(
        board.legal_moves, key=lambda m: -(_PIECE_CP.get(board.piece_type_at(m.to_square) or 0, 0))
    )
    best = ordered[0]

    def negamax(depth: int, alpha: int, beta: int, ply: int) -> int:
        if time.perf_counter() >= deadline:
            raise _Deadline
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        if depth <= 0:
            return _material(board)
        moves = sorted(
            board.legal_moves,
            key=lambda m: -(_PIECE_CP.get(board.piece_type_at(m.to_square) or 0, 0)),
        )
        value = -MATE - 1
        for move in moves:
            board.push(move)
            score = -negamax(depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

    try:
        for depth in range(1, 6):
            alpha = -MATE - 1
            current = best
            for move in ordered:
                board.push(move)
                score = -negamax(depth - 1, -MATE - 1, -alpha, 1)
                board.pop()
                if score > alpha:
                    alpha = score
                    current = move
            best = current
            ordered.remove(best)
            ordered.insert(0, best)
            if alpha >= MATE - 64:
                break
    except _Deadline:
        while len(board.move_stack) > depth0:  # unwind the interrupted line
            board.pop()
    return best


# ----------------------------------------------------------------------------- game state


def reset_game() -> None:
    global _LAST_FEN, _LAST_REPLY
    _HISTORY_FENS.clear()
    _HISTORY_KEYS.clear()
    _LAST_FEN = None
    _LAST_REPLY = None
    if _ENGINE_READY.is_set():
        SEARCHER.clear()


def _update_history(board: chess.Board) -> None:
    """Keep the history consistent with the game we are apparently in."""
    global _LAST_FEN
    if _LAST_FEN is not None and _LAST_REPLY is not None:
        prev = chess.Board(_LAST_FEN)
        prev.push(_LAST_REPLY)
        reachable = False
        for reply in prev.legal_moves:
            prev.push(reply)
            if prev.board_fen() == board.board_fen() and prev.turn == board.turn:
                reachable = True
            prev.pop()
            if reachable:
                break
        if not reachable:
            _HISTORY_FENS.clear()  # a new game or an unexpected position
            _HISTORY_KEYS.clear()
    _LAST_FEN = board.fen()


def _history_keys() -> list[int]:
    """Zobrist keys (cboard's) of the game history; converts FENs lazily once the engine exists."""
    while len(_HISTORY_KEYS) < len(_HISTORY_FENS):
        _HISTORY_KEYS.append(
            int(cboard.from_board(chess.Board(_HISTORY_FENS[len(_HISTORY_KEYS)]))[cboard.HASH])
        )
    return _HISTORY_KEYS


def _engine_move(board: chess.Board, time_left_ms: int) -> tuple[chess.Move | None, dict[str, int]]:
    SEARCHER.set_position(board, _history_keys())
    budget = budget_seconds(time_left_ms)
    if budget <= 0.0:
        move_int, _score, _depth, _pv, stats = SEARCHER.search(max_depth=2, node_limit=400)
    else:
        move_int, _score, _depth, _pv, stats = SEARCHER.search(time_budget=budget)
    uci = cboard.move_to_uci(move_int) if move_int else ""
    move = chess.Move.from_uci(uci) if uci else None
    if move is None or move not in board.legal_moves:
        print(f"[21_nnue] search returned {uci!r}, using fallback", file=sys.stderr)
        return None, {"nodes": 0, "depth": 0}
    return move, {"nodes": int(stats["nodes"]), "depth": int(stats["depth"])}


def get_move(fen: str, time_left_ms: int) -> str:
    global _LAST_REPLY
    t0 = time.perf_counter()
    board = chess.Board(fen)
    fallback = _fallback_move(board)
    if fallback is None:
        return "0000"  # checkmate or stalemate: nothing legal exists
    move: chess.Move | None = None
    stats = {"nodes": 0, "depth": 0}
    try:
        _update_history(board)
        if _ENGINE_READY.is_set() and _ENGINE_ERROR is None:
            move, stats = _engine_move(board, time_left_ms)
        else:
            _STATS["fallback"] += 1
            move = _python_search(board, t0 + max(MIN_BUDGET_S, budget_seconds(time_left_ms) * 0.5))
    except Exception as exc:  # never lose on an exception: play the fallback
        print(f"[21_nnue] search failed: {exc!r}", file=sys.stderr)
        move = None
    if move is None or move not in board.legal_moves:
        move = fallback
    # remember the position we saw and the one our move produces
    _HISTORY_FENS.append(board.fen())
    board.push(move)
    _HISTORY_FENS.append(board.fen())
    _LAST_REPLY = move
    elapsed = time.perf_counter() - t0
    _STATS["moves"] += 1
    _STATS["nodes"] += stats.get("nodes", 0)
    _STATS["time"] += elapsed
    _STATS["max_time"] = max(_STATS["max_time"], elapsed)
    _STATS["depth_sum"] += stats.get("depth", 0)
    return move.uci()


# ----------------------------------------------------------------------------- start-up


def _build_engine() -> None:
    """Compile every numba path (first calls trigger compilation) and warm the searcher."""
    global _ENGINE_ERROR
    try:
        fens = [
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/P4k2/8/8/8/8/1p3K2/8 w - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            SEARCHER.set_position(board, [])
            SEARCHER.search(max_depth=3, node_limit=3000)
        SEARCHER.clear()
    except BaseException as exc:
        _ENGINE_ERROR = exc
        print(f"[21_nnue] engine build failed: {exc!r}", file=sys.stderr)
    finally:
        wall = time.perf_counter() - _INIT_START
        print(
            f"[21_nnue] engine ready after {wall:.1f}s wall, {time.process_time():.1f}s cpu, "
            f"nnue={'yes' if NET else 'no'}",
            file=sys.stderr,
        )
        _ENGINE_READY.set()


def wait_ready(timeout: float | None = None) -> bool:
    """Block until the compiled engine is usable (tests and tools; the platform never needs it)."""
    return _ENGINE_READY.wait(timeout)


_BUILD_THREAD = threading.Thread(target=_build_engine, name="nnue21-build", daemon=True)
_BUILD_THREAD.start()
_ENGINE_READY.wait(max(0.0, INIT_WAIT_S - (time.perf_counter() - _INIT_START)))
if not _ENGINE_READY.is_set():
    print("[21_nnue] engine still compiling; early moves use the python fallback", file=sys.stderr)
