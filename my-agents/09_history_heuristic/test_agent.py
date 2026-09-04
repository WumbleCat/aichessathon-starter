"""Tests for the history heuristic agent, one per bullet in the spec's Tests section.

Run from the repo root:

    uv run python -m unittest my-agents/09_history_heuristic/test_agent.py
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import chess

_AGENT_PATH = Path(__file__).with_name("agent.py")
_SPEC = importlib.util.spec_from_file_location("history_agent", _AGENT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
agent = importlib.util.module_from_spec(_SPEC)
sys.modules["history_agent"] = agent
_SPEC.loader.exec_module(agent)

# Representative middlegame positions: enough quiet moves for history to matter.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "2r3k1/pp3ppp/2n5/3p4/3P4/2N2N2/PP3PPP/2R3K1 b - - 0 20",
    "8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 40",
]


def fixed_depth(board: chess.Board, depth: int, **flags: bool) -> tuple[chess.Move, int, int]:
    """Search to a fixed depth with a fresh searcher; return (move, score, nodes)."""
    searcher = agent.Searcher(**flags)
    searcher.new_search(float("inf"))
    move, score = searcher.search_root(board, depth)
    return move, score, searcher.nodes


class HistoryTableTest(unittest.TestCase):
    def test_quiet_cutoff_increases_history_value(self) -> None:
        history = agent.HistoryTable()
        move = chess.Move.from_uci("g1f3")
        self.assertEqual(history.get(chess.WHITE, move), 0)
        history.update(chess.WHITE, move, [], depth=3)
        self.assertEqual(history.get(chess.WHITE, move), 9)
        # The other colour's entry for the same squares is untouched.
        self.assertEqual(history.get(chess.BLACK, move), 0)

    def test_failed_quiet_moves_are_penalised(self) -> None:
        history = agent.HistoryTable()
        cutoff = chess.Move.from_uci("g1f3")
        tried = [chess.Move.from_uci("a2a3"), chess.Move.from_uci("h2h3")]
        history.update(chess.WHITE, cutoff, tried, depth=2)
        self.assertEqual(history.get(chess.WHITE, cutoff), 4)
        for move in tried:
            self.assertEqual(history.get(chess.WHITE, move), -4)

    def test_higher_history_quiet_move_sorts_first(self) -> None:
        board = chess.Board()
        history = agent.HistoryTable()
        late = chess.Move.from_uci("a2a3")
        history.update(chess.WHITE, late, [], depth=5)
        moves = agent.ordered_moves(board, history=history)
        self.assertEqual(moves[0], late)
        # Without history the same move is not special.
        plain = agent.ordered_moves(board)
        self.assertNotEqual(plain[0], late)

    def test_captures_remain_governed_by_capture_ordering(self) -> None:
        # White can capture the queen with a pawn (e4xd5) or play a quiet move whose
        # history score is huge. The capture must still be searched first.
        board = chess.Board("rnb1kbnr/ppp1pppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        history = agent.HistoryTable()
        quiet = chess.Move.from_uci("g1f3")
        history.table[history.index(chess.WHITE, quiet)] = agent.HISTORY_MAX - 1
        moves = agent.ordered_moves(board, history=history)
        self.assertEqual(moves[0], chess.Move.from_uci("e4d5"))
        priorities = [agent.move_priority(board, m, None, agent.NO_KILLERS, history) for m in moves]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
        # Every capture outranks every quiet move regardless of history.
        scored = list(zip(moves, priorities, strict=True))
        capture_min = min(p for m, p in scored if board.is_capture(m))
        quiet_max = max(p for m, p in scored if agent.is_quiet(board, m))
        self.assertGreater(capture_min, quiet_max)

    def test_killers_outrank_history(self) -> None:
        board = chess.Board()
        history = agent.HistoryTable()
        historic = chess.Move.from_uci("a2a3")
        history.table[history.index(chess.WHITE, historic)] = agent.HISTORY_MAX - 1
        killer = chess.Move.from_uci("h2h3")
        moves = agent.ordered_moves(board, killers=(killer, None), history=history)
        self.assertEqual(moves[0], killer)
        self.assertEqual(moves[1], historic)

    def test_history_table_can_be_reset(self) -> None:
        history = agent.HistoryTable()
        history.update(chess.BLACK, chess.Move.from_uci("g8f6"), [], depth=4)
        self.assertTrue(any(history.table))
        history.reset()
        self.assertFalse(any(history.table))
        self.assertEqual(len(history.table), agent.HistoryTable.SIZE)

    def test_history_is_bounded(self) -> None:
        history = agent.HistoryTable()
        move = chess.Move.from_uci("g1f3")
        loser = chess.Move.from_uci("a2a3")
        for _ in range(10_000):
            history.update(chess.WHITE, move, [loser], depth=8)
        self.assertLess(abs(history.get(chess.WHITE, move)), agent.HISTORY_MAX)
        self.assertLess(abs(history.get(chess.WHITE, loser)), agent.HISTORY_MAX)
        self.assertTrue(all(abs(v) < agent.HISTORY_MAX for v in history.table))

    def test_aging_halves_towards_zero(self) -> None:
        history = agent.HistoryTable()
        up = chess.Move.from_uci("g1f3")
        down = chess.Move.from_uci("a2a3")
        history.table[history.index(chess.WHITE, up)] = 9
        history.table[history.index(chess.WHITE, down)] = -9
        history.age()
        self.assertEqual(history.get(chess.WHITE, up), 4)
        self.assertEqual(history.get(chess.WHITE, down), -4)


class SearchTest(unittest.TestCase):
    def test_same_search_result_as_alpha_beta_without_history(self) -> None:
        # With the TT off the search is pure alpha-beta: move ordering can only change
        # the node count, never the minimax score.
        for fen in POSITIONS:
            for depth in (2, 3):
                with self.subTest(fen=fen, depth=depth):
                    _, with_history, _ = fixed_depth(
                        chess.Board(fen), depth, use_history=True, use_tt=False
                    )
                    _, without_history, _ = fixed_depth(
                        chess.Board(fen), depth, use_history=False, use_tt=False
                    )
                    self.assertEqual(with_history, without_history)

    def test_history_reduces_nodes_on_representative_positions(self) -> None:
        total_with = total_without = 0
        for fen in POSITIONS:
            _, _, with_nodes = fixed_depth(chess.Board(fen), 4, use_history=True, use_tt=False)
            _, _, without_nodes = fixed_depth(
                chess.Board(fen), 4, use_history=False, use_tt=False
            )
            total_with += with_nodes
            total_without += without_nodes
        print(f"\nnodes at depth 4 over {len(POSITIONS)} positions: "
              f"history {total_with}, no history {total_without}")
        self.assertLess(total_with, total_without)

    def test_search_is_deterministic(self) -> None:
        board = chess.Board(POSITIONS[2])
        first = fixed_depth(board, 3)
        second = fixed_depth(board, 3)
        self.assertEqual(first, second)

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        move, score, _ = fixed_depth(board, 2)
        self.assertEqual(move, chess.Move.from_uci("a1a8"))
        self.assertGreaterEqual(score, agent.MATE_SCORE - agent.MAX_PLY)

    def test_prefers_shorter_mate(self) -> None:
        # Mate in one is available; a deeper search must still pick it.
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        move, _, _ = fixed_depth(board, 4)
        self.assertEqual(move, chess.Move.from_uci("a1a8"))

    def test_get_move_returns_legal_uci(self) -> None:
        for fen in POSITIONS:
            board = chess.Board(fen)
            uci = agent.get_move(fen, 2_000)
            self.assertIn(chess.Move.from_uci(uci), board.legal_moves)

    def test_get_move_when_only_one_legal_move(self) -> None:
        fen = "7k/8/5K2/8/8/8/8/6R1 b - - 0 1"  # black king boxed in, single legal move
        board = chess.Board(fen)
        uci = agent.get_move(fen, 1_000)
        self.assertIn(chess.Move.from_uci(uci), board.legal_moves)

    def test_respects_time_budget(self) -> None:
        import time

        start = time.monotonic()
        agent.get_move(POSITIONS[2], 3_000)
        elapsed_ms = (time.monotonic() - start) * 1000
        # Budget is 3000 // 30 = 100 ms, plus one over-run of at most 1024 nodes.
        self.assertLess(elapsed_ms, 600)


if __name__ == "__main__":
    unittest.main()
