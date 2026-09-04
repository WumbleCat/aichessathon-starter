"""Tests for the iterative deepening agent.

Run from the repository root with the project's interpreter:

    uv run python -m unittest discover -s my-agents/05_iterative_deepening/tests -v
"""

import sys
import time
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent

MIDDLEGAME = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"  # Re8#
MATE_IN_TWO = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1"  # Nf6+ gxf6 Bxf7#


class EvaluationTests(unittest.TestCase):
    def test_extra_queen_is_about_nine_pawns(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        self.assertGreater(agent.evaluate(board), 800)
        self.assertLess(agent.evaluate(board), 1000)

    def test_extra_pawn_is_about_one_pawn(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        self.assertGreater(agent.evaluate(board), 50)
        self.assertLess(agent.evaluate(board), 200)

    def test_centralised_knight_beats_corner_knight(self) -> None:
        centre = chess.Board("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
        corner = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
        self.assertGreater(agent.evaluate(centre), agent.evaluate(corner))

    def test_mirrored_positions_score_opposite(self) -> None:
        board = chess.Board(MIDDLEGAME)
        mirrored = board.mirror()
        # From the mover's point of view the two positions are identical.
        self.assertEqual(agent.evaluate(board), agent.evaluate(mirrored))

    def test_passed_pawn_gets_bonus(self) -> None:
        # Same material either way: only the black pawn's file changes.
        passed = chess.Board("4k3/8/p7/8/3P4/8/8/4K3 w - - 0 1")
        blocked = chess.Board("4k3/8/4p3/8/3P4/8/8/4K3 w - - 0 1")
        self.assertGreater(agent.evaluate(passed), agent.evaluate(blocked))
        self.assertGreater(agent._pawn_structure(passed, chess.WHITE), 0)
        self.assertEqual(agent._pawn_structure(blocked, chess.WHITE), -agent.ISOLATED_PAWN_PENALTY)

    def test_doubled_pawns_are_penalised(self) -> None:
        healthy = chess.Board("4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1")
        doubled = chess.Board("4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1")
        self.assertGreater(agent.evaluate(healthy), agent.evaluate(doubled))

    def test_bishop_pair_gets_bonus(self) -> None:
        pair = chess.Board("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
        knight_and_bishop = chess.Board("4k3/8/8/8/8/8/8/2N1KB2 w - - 0 1")
        material_difference = agent.PIECE_VALUE[chess.BISHOP] - agent.PIECE_VALUE[chess.KNIGHT]
        self.assertGreater(
            agent.evaluate(pair) - agent.evaluate(knight_and_bishop), material_difference
        )


class SearchTests(unittest.TestCase):
    def test_depth_one_result_exists(self) -> None:
        result = agent.search(chess.Board(MIDDLEGAME), max_depth=1)
        self.assertEqual(result.depth, 1)
        self.assertEqual(result.completed_depths, [1])
        self.assertIn(result.move, chess.Board(MIDDLEGAME).legal_moves)

    def test_depth_increases_sequentially(self) -> None:
        result = agent.search(chess.Board(MIDDLEGAME), max_depth=4)
        self.assertEqual(result.completed_depths, [1, 2, 3, 4])
        self.assertEqual(result.depth, 4)
        self.assertGreater(result.nodes, 0)

    def test_best_move_is_legal_in_every_iteration(self) -> None:
        for fen in (chess.STARTING_FEN, MIDDLEGAME, MATE_IN_TWO):
            board = chess.Board(fen)
            for depth in range(1, 4):
                result = agent.search(board, max_depth=depth)
                self.assertIn(result.move, board.legal_moves)
                self.assertEqual(board.fen(), fen, "search must restore the board")

    def test_pv_is_a_legal_line_starting_with_best_move(self) -> None:
        board = chess.Board(MIDDLEGAME)
        result = agent.search(board, max_depth=3)
        self.assertTrue(result.pv)
        self.assertEqual(result.pv[0], result.move)
        replay = chess.Board(MIDDLEGAME)
        for move in result.pv:
            self.assertIn(move, replay.legal_moves)
            replay.push(move)

    def test_timed_search_stops_after_a_completed_iteration(self) -> None:
        board = chess.Board(MIDDLEGAME)
        started = time.monotonic()
        result = agent.search(board, time_limit_s=0.3)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.3 + 0.25)  # abort granularity is a few hundred nodes
        self.assertEqual(board.fen(), MIDDLEGAME, "search must restore the board")
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.completed_depths, list(range(1, result.depth + 1)))
        self.assertEqual(result.pv[0], result.move)

    def test_aborted_iteration_restores_board_and_keeps_last_depth(self) -> None:
        # Force the abort path: allow depth 2 to start with a budget that is already gone.
        board = chess.Board(MIDDLEGAME)
        fraction = agent.NEXT_ITERATION_FRACTION
        agent.NEXT_ITERATION_FRACTION = 1e9
        try:
            result = agent.search(board, time_limit_s=0.001)
        finally:
            agent.NEXT_ITERATION_FRACTION = fraction
        self.assertTrue(result.aborted)
        self.assertEqual(result.depth, 1)
        self.assertEqual(board.fen(), MIDDLEGAME, "abort must unwind pushed moves")
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.move, agent.search(chess.Board(MIDDLEGAME), max_depth=1).move)

    def test_tiny_budget_still_returns_depth_one(self) -> None:
        board = chess.Board(MIDDLEGAME)
        result = agent.search(board, time_limit_s=0.0)
        self.assertEqual(result.depth, 1)
        self.assertIn(result.move, board.legal_moves)

    def test_timed_result_matches_fixed_depth_with_enough_time(self) -> None:
        board = chess.Board(MIDDLEGAME)
        fixed = agent.search(board, max_depth=3)
        timed = agent.search(board, max_depth=3, time_limit_s=60.0)
        self.assertEqual(timed.depth, 3)
        self.assertEqual(fixed.move, timed.move)
        self.assertEqual(fixed.score, timed.score)

    def test_previous_best_move_is_searched_first(self) -> None:
        board = chess.Board(MIDDLEGAME)
        first = chess.Move.from_uci("h2h3")  # a quiet move that ordering would never lead with
        moves = agent.ordered_moves(board, first=first)
        self.assertEqual(moves[0], first)
        self.assertEqual(sorted(m.uci() for m in moves), sorted(m.uci() for m in board.legal_moves))

    def test_previous_iteration_reduces_nodes(self) -> None:
        """Iterating 1..4 must not cost more than a cold depth-4 search with a bad first move."""
        board = chess.Board(MIDDLEGAME)
        iterative = agent.search(board, max_depth=4)
        cold = agent.Searcher()
        worst_first = agent.ordered_moves(board)[-1]
        cold.search_root(board, 4, first=worst_first)
        self.assertLess(iterative.nodes, cold.nodes)

    def test_mate_in_one_is_found_and_deepening_stops(self) -> None:
        board = chess.Board(MATE_IN_ONE)
        result = agent.search(board, max_depth=6)
        self.assertEqual(result.move.uci(), "e1e8")
        self.assertTrue(agent.is_mate_score(result.score))
        self.assertEqual(agent.format_score(result.score), "#1")
        self.assertLess(result.depth, 6, "deepening should stop once mate is proven")

    def test_mate_in_two_is_found(self) -> None:
        board = chess.Board(MATE_IN_TWO)
        result = agent.search(board, max_depth=3)
        self.assertEqual(result.move.uci(), "d5f6")
        self.assertEqual(agent.format_score(result.score), "#2")

    def test_quiescence_does_not_grab_a_defended_piece(self) -> None:
        # The white queen is attacked by a rook that a pawn defends. Taking the rook
        # loses the queen; a depth-1 search with quiescence must see the recapture.
        board = chess.Board("4k3/8/3p4/4r3/8/8/8/K3Q3 w - - 0 1")
        result = agent.search(board, max_depth=1)
        self.assertNotEqual(result.move.uci(), "e1e5")
        self.assertGreater(result.score, -200)

    def test_in_check_quiescence_searches_evasions(self) -> None:
        searcher = agent.Searcher()
        board = chess.Board("4k3/8/8/8/8/8/8/r3K3 w - - 0 1")  # white in check from the rook
        score = searcher.quiescence(board, -agent.INFINITY, agent.INFINITY, 0)
        self.assertFalse(agent.is_mate_score(score))
        self.assertLess(score, 0)


class InterfaceTests(unittest.TestCase):
    def test_get_move_returns_legal_uci(self) -> None:
        for fen in (chess.STARTING_FEN, MIDDLEGAME, MATE_IN_ONE):
            board = chess.Board(fen)
            uci = agent.get_move(fen, 2_000)
            self.assertIn(chess.Move.from_uci(uci), board.legal_moves)

    def test_get_move_respects_a_short_clock(self) -> None:
        started = time.monotonic()
        uci = agent.get_move(MIDDLEGAME, 300)
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertIn(chess.Move.from_uci(uci), chess.Board(MIDDLEGAME).legal_moves)

    def test_promotion_is_returned_with_suffix(self) -> None:
        board = chess.Board("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1")
        uci = agent.get_move(board.fen(), 1_000)
        self.assertEqual(uci, "e7e8q")

    def test_move_budget_never_exceeds_clock(self) -> None:
        for time_left in (1, 50, 200, 5_000, 120_000):
            budget = agent.move_budget_ms(time_left)
            self.assertGreaterEqual(budget, 1)
            self.assertLessEqual(budget, max(1, time_left - agent.SAFETY_MARGIN_MS))
            self.assertLessEqual(budget, agent.MAX_BUDGET_MS)


if __name__ == "__main__":
    unittest.main()
