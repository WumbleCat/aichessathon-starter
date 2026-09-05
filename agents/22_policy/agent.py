"""22_policy: alpha-beta search whose root/shallow move ordering comes from a policy network.

Entry point: ``get_move(fen, time_left_ms) -> uci``.

The network (``models/policy.npz``) is a small residual CNN trained from random initialisation
by this project on positions labelled by this project's own search; see ``training/``. If the
weights are missing the engine still plays, using hand-crafted ordering only.
"""

from __future__ import annotations

import os
import sys
import time

# one core: a multi-threaded BLAS only loses time (and must be set before numpy is imported)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import chess  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pn_search import Searcher  # noqa: E402

MAX_DEPTH = 40
MOVES_TO_GO = 28
MAX_BUDGET_S = 6.0
MIN_BUDGET_S = 0.01
OVERHEAD_MS = 60  # round trip through the runner, FEN parsing, clock granularity
PANIC_MS = 200  # below this, do not search at all

POLICY_MIN_DEPTH = int(os.environ.get("PN_POLICY_MIN_DEPTH", "4"))
POLICY_ROOT = os.environ.get("PN_POLICY_ROOT", "1") == "1"
POLICY_LMR = os.environ.get("PN_POLICY_LMR", "1") == "1"
USE_POLICY = os.environ.get("PN_USE_POLICY", "1") == "1"
SEARCHLESS = os.environ.get("PN_SEARCHLESS", "0") == "1"

prior = None
policy_net = None
if USE_POLICY:
    try:
        from pn_policy import load_policy

        policy_net = load_policy(os.environ.get("PN_MODEL_PATH", os.path.join(HERE, "models", "policy.npz")))
        if policy_net is not None:
            prior = policy_net.prior
    except Exception as error:  # the engine must still play without the network
        print(f"policy unavailable: {error!r}")
        prior = None

searcher = Searcher(prior=prior, policy_min_depth=POLICY_MIN_DEPTH, policy_root=POLICY_ROOT, policy_lmr=POLICY_LMR)

_last_fullmove = -1
_last_key: object = None


def budget_seconds(time_left_ms: int, board: chess.Board) -> float:
    usable = max(time_left_ms - OVERHEAD_MS, 0)
    per_move = usable / MOVES_TO_GO
    if time_left_ms < 3_000:
        per_move = usable / (2 * MOVES_TO_GO)
    return max(MIN_BUDGET_S, min(MAX_BUDGET_S, per_move / 1000.0))


def fallback_move(board: chess.Board) -> chess.Move:
    """A legal move chosen without searching: best capture by victim value, else first."""
    best: chess.Move | None = None
    best_value = -1
    for move in board.legal_moves:
        victim = board.piece_type_at(move.to_square)
        value = 0 if victim is None else int(victim)
        if move.promotion == chess.QUEEN:
            value += 10
        if value > best_value:
            best, best_value = move, value
    assert best is not None, "no legal moves"
    return best


def _track_game(board: chess.Board) -> None:
    """Keep repetition memory across calls; reset it when a new game appears to have begun."""
    global _last_fullmove, _last_key
    if board.fullmove_number < _last_fullmove or _last_fullmove < 0:
        searcher.new_game()
    elif _last_fullmove >= 0 and board.fullmove_number - _last_fullmove > 3:
        searcher.new_game()
    _last_fullmove = board.fullmove_number
    _last_key = board._transposition_key()
    searcher.remember_position(board)


def get_move(fen: str, time_left_ms: int) -> str:
    started = time.perf_counter()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("no legal moves in position")
    fallback = fallback_move(board)
    if len(legal) == 1:
        return legal[0].uci()
    if time_left_ms < PANIC_MS:
        return fallback.uci()

    _track_game(board)

    if SEARCHLESS and policy_net is not None:
        p = policy_net.prior(board)
        move = max(legal, key=lambda m: p.get(m, 0.0))
        return move.uci()

    move: chess.Move | None = None
    try:
        result = searcher.search(board, MAX_DEPTH, time_budget=budget_seconds(time_left_ms, board))
        move = result.move
        st = result.stats
        total = st.nodes
        nps = int(total / result.elapsed) if result.elapsed > 0 else 0
        print(
            f"depth {result.depth} sel {st.seldepth} score {result.score} nodes {total} "
            f"q {st.qnodes} tt {st.tt_hits} pol {st.policy_calls} nps {nps} "
            f"t {time.perf_counter() - started:.2f}s pv {' '.join(m.uci() for m in result.pv[:6])}",
            flush=True,
        )
    except Exception as error:  # a crash loses; a legal move does not
        print(f"search failed: {error!r}")

    if move is None or move not in board.legal_moves:
        move = fallback
    return move.uci()
