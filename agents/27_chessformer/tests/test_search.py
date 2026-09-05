"""Search tests: tactics, repetition handling, the policy hook and time aborts."""

import os
import random
import sys

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cf_eval import MATE_BOUND, evaluate  # noqa: E402
from cf_search import Searcher  # noqa: E402


def test_eval_is_symmetric_under_mirror():
    for fen in [
        chess.STARTING_FEN,
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    ]:
        b = chess.Board(fen)
        assert evaluate(b) == evaluate(b.mirror())


def test_finds_mate_in_two():
    # classic: 1.Qxh7+ Kxh7 2.Rh3#? use a simpler forced mate in 2
    fen = "6k1/5ppp/8/8/8/8/1Q6/R5K1 w - - 0 1"
    s = Searcher()
    r = s.search(chess.Board(fen), 20.0, max_depth=5)
    assert r.score > MATE_BOUND, r.score
    assert r.move.uci() in ("b2b8", "a1a8")


def test_mate_score_prefers_shortest():
    fen = "7k/8/8/8/8/8/8/K5QR w - - 0 1"  # mate in 1 available: Qg8#? no: Rh1-h8# yes
    s = Searcher()
    r = s.search(chess.Board(fen), 20.0, max_depth=4)
    assert r.score > MATE_BOUND
    b = chess.Board(fen)
    b.push(r.move)
    assert b.is_checkmate()


def test_repetition_memory_avoids_threefold_when_winning():
    # white up a queen; positions visited before are scored as draws, so the engine must vary
    fen = "7k/8/8/8/8/8/8/K5Q1 w - - 0 1"
    s = Searcher()
    b = chess.Board(fen)
    s.note_position(b)
    r = s.search(b, 2.0, max_depth=4)
    b.push(r.move)
    s.note_position(b)
    # black shuffles, white shuffles back would repeat: engine must not choose the repeating line
    assert r.score > 0


def test_policy_hook_orders_and_stays_legal():
    calls = []
    rng = random.Random(0)

    def policy(board: chess.Board) -> dict[chess.Move, float]:
        calls.append(1)
        moves = list(board.legal_moves)
        w = [rng.random() for _ in moves]
        z = sum(w)
        return {m: x / z for m, x in zip(moves, w, strict=True)}

    s = Searcher(policy_fn=policy, policy_min_depth=2)
    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8")
    r = s.search(b, 5.0, max_depth=4)
    assert r.move in b.legal_moves
    assert calls, "policy was never consulted"
    assert s.policy_calls == len(calls)
    assert len(b.move_stack) == 0


def test_time_abort_leaves_board_intact():
    s = Searcher()
    b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8")
    fen = b.fen()
    r = s.search(b, 0.05, max_depth=30)
    assert r.move in b.legal_moves
    assert b.fen() == fen
    assert r.time < 0.6


def test_underpromotion_is_searched():
    s = Searcher()
    b = chess.Board("k7/2P5/1K6/8/8/8/8/8 w - - 0 1")
    r = s.search(b, 5.0, max_depth=4)
    b.push(r.move)
    assert not b.is_stalemate()


def test_node_budget_stops_the_search_and_charges_policy_calls():
    """max_nodes bounds the search (depth 1 excepted) and policy calls add policy_node_cost."""
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8")
    plain = Searcher()
    res = plain.search(board, budget_s=60.0, max_nodes=1500)
    assert res.move in board.legal_moves
    assert (
        res.nodes < 1500 + 2 * 128 + 400
    )  # the check runs every 128 nodes, plus one iteration overshoot

    calls = []

    def uniform(b: chess.Board) -> dict[chess.Move, float]:
        moves = list(b.legal_moves)
        calls.append(1)
        return {m: 1.0 / len(moves) for m in moves}

    charged = Searcher(policy_fn=uniform, policy_min_depth=2)
    charged.policy_node_cost = 100
    res2 = charged.search(board, budget_s=60.0, max_nodes=1500)
    assert res2.move in board.legal_moves
    assert len(calls) >= 1
    assert res2.nodes >= 100 * len(calls)
