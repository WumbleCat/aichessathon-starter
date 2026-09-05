"""AI Chessathon entry point: self-trained NNUE evaluation inside an own PVS search.

    get_move(fen, time_left_ms) -> uci

Pipeline per call: python-chess parses the FEN and gives us a guaranteed-legal fallback move,
the position is copied into our numba bitboard engine (cboard), the search (csearch) runs
with the NNUE evaluator (nnue) until its time budget is spent, and the chosen move is
validated against python-chess before it is returned.  Nothing here reads the network or
calls an external engine; the weights in ``weights/`` were trained by this project
(see training/ and models/PROVENANCE.md).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# numba may cache compiled code; keep that in a writable temp dir (the platform's /tmp)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "nnue21_numba_cache"))

import chess  # noqa: E402

import cboard  # noqa: E402
import csearch  # noqa: E402
import nnue  # noqa: E402

_INIT_START = time.perf_counter()

WEIGHTS = nnue.default_weights_path()
NET = nnue.load_net(WEIGHTS) if os.path.exists(WEIGHTS) else None
SEARCHER = csearch.Searcher(NET, use_nnue=NET is not None)

# Positions this process has been asked about (and the ones our replies produced), as
# zobrist keys, so the search can see game-level repetitions.  The harness only sends FENs.
_HISTORY: list[int] = []
_LAST_FEN: str | None = None
_LAST_REPLY: chess.Move | None = None
_STATS: dict[str, float] = {"moves": 0, "nodes": 0, "time": 0.0, "max_time": 0.0, "depth_sum": 0}

MIN_BUDGET_S = 0.004
OVERHEAD_S = 0.015  # python-chess parsing, validation, protocol


def budget_seconds(time_left_ms: int) -> float:
    """Seconds the search may use.  Assumes 0.5 s increments and about 30 more moves."""
    t = time_left_ms / 1000.0
    if t <= 0.15:
        return 0.0  # node-limited emergency search
    if t < 2.0:
        return max(MIN_BUDGET_S, t * 0.04)
    b = t / 30.0 + 0.25
    return min(b, t * 0.2) - OVERHEAD_S


def _fallback_move(board: chess.Board) -> chess.Move:
    """A legal move that exists before any expensive work: best capture by victim value."""
    best: chess.Move | None = None
    best_value = -1
    for move in board.legal_moves:
        value = 0
        victim = board.piece_type_at(move.to_square)
        if victim is not None:
            value = int(csearch.PIECE_VALUE[victim])
        if move.promotion == chess.QUEEN:
            value += 800
        if value > best_value:
            best_value = value
            best = move
    assert best is not None
    return best


def reset_game() -> None:
    global _LAST_FEN, _LAST_REPLY
    _HISTORY.clear()
    _LAST_FEN = None
    _LAST_REPLY = None
    SEARCHER.clear()


def _update_history(board: chess.Board) -> None:
    """Keep the key list consistent with the game we are apparently in."""
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
            _HISTORY.clear()  # a new game or an unexpected position
    _LAST_FEN = board.fen()


def get_move(fen: str, time_left_ms: int) -> str:
    global _LAST_REPLY
    t0 = time.perf_counter()
    board = chess.Board(fen)
    fallback = _fallback_move(board)
    try:
        _update_history(board)
        SEARCHER.set_position(board, _HISTORY)
        budget = budget_seconds(time_left_ms)
        if budget <= 0.0:
            move_int, score, depth, pv, stats = SEARCHER.search(max_depth=2, node_limit=400)
        else:
            move_int, score, depth, pv, stats = SEARCHER.search(time_budget=budget)
        uci = cboard.move_to_uci(move_int) if move_int else ""
        move = chess.Move.from_uci(uci) if uci else None
        if move is None or move not in board.legal_moves:
            print(f"[21_nnue] search returned {uci!r}, using fallback", file=sys.stderr)
            move = fallback
            stats = {"nodes": 0, "depth": 0}
    except Exception as exc:  # never lose on an exception: play the fallback
        print(f"[21_nnue] search failed: {exc!r}", file=sys.stderr)
        move = fallback
        stats = {"nodes": 0, "depth": 0}
    # remember the position we saw and the one our move produces
    key_before = int(SEARCHER.P[cboard.HASH])
    _HISTORY.append(key_before)
    board.push(move)
    _HISTORY.append(int(cboard.from_board(board)[cboard.HASH]))
    _LAST_REPLY = move
    elapsed = time.perf_counter() - t0
    _STATS["moves"] += 1
    _STATS["nodes"] += stats.get("nodes", 0)
    _STATS["time"] += elapsed
    _STATS["max_time"] = max(_STATS["max_time"], elapsed)
    _STATS["depth_sum"] += stats.get("depth", 0)
    return move.uci()


def _warm_up() -> None:
    """Compile every numba path inside the init budget, not on the clock."""
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
    print(f"[21_nnue] init {time.perf_counter() - _INIT_START:.1f}s, nnue={'yes' if NET else 'no'}", file=sys.stderr)


_warm_up()
