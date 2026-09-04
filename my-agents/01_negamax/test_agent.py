"""Tests for the stage 01 negamax bot.

Run from the repository root:

    uv run python -m unittest discover -s my-agents/01_negamax -v
"""

import sys
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent

# A rook on the open a-file mates on a8 in one; the king on g1 covers nothing relevant.
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
# Black's queen sits on h4 with no defender, and White's knight on f3 attacks it.
HANGING_QUEEN = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
# Black to move, not in check, no legal move.
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
# Black to move, in check from g7, no escape.
CHECKMATED = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
# White mates on the spot with Qb8 or in two after a quiet move; the search must prefer Qb8.
MATE_CHOICE = "6k1/8/6K1/8/8/8/8/1Q6 w - - 0 1"


class NegamaxTests(unittest.TestCase):
    def setUp(self) -> None:
        agent.reset_game()

    def test_starting_position_returns_a_legal_move(self) -> None:
        board = chess.Board()
        for depth in (1, 2, 3):
            move, _ = agent.search_root(board, depth)
            self.assertIsNotNone(move)
            self.assertIn(move, board.legal_moves)
        uci = agent.get_move(chess.STARTING_FEN, 120_000)
        self.assertIn(chess.Move.from_uci(uci), chess.Board().legal_moves)

    def test_one_move_checkmate_is_found(self) -> None:
        board = chess.Board(MATE_IN_ONE)
        for depth in (1, 2, 3):
            move, score = agent.search_root(board, depth)
            self.assertEqual(move, chess.Move.from_uci("a1a8"), f"depth {depth}")
            self.assertEqual(score, agent.MATE_SCORE - 1, f"depth {depth}")
        self.assertEqual(agent.get_move(MATE_IN_ONE, 120_000), "a1a8")

    def test_faster_mate_is_preferred(self) -> None:
        board = chess.Board(MATE_CHOICE)
        move, score = agent.search_root(board, 3)
        self.assertEqual(move, chess.Move.from_uci("b1b8"))
        self.assertEqual(score, agent.MATE_SCORE - 1)

    def test_hanging_queen_is_captured(self) -> None:
        board = chess.Board(HANGING_QUEEN)
        for depth in (1, 2, 3):
            move, _ = agent.search_root(board, depth)
            self.assertEqual(move, chess.Move.from_uci("f3h4"), f"depth {depth}")

    def test_stalemate_scores_zero(self) -> None:
        board = chess.Board(STALEMATE)
        self.assertTrue(board.is_stalemate())
        for depth in (0, 1, 3):
            self.assertEqual(agent.negamax(board, depth), 0)
        move, score = agent.search_root(board, 3)
        self.assertIsNone(move)
        self.assertEqual(score, 0)

    def test_checkmate_scores_large_negative_for_side_to_move(self) -> None:
        board = chess.Board(CHECKMATED)
        self.assertTrue(board.is_checkmate())
        for depth in (0, 1, 3):
            self.assertEqual(agent.negamax(board, depth), -agent.MATE_SCORE)
        move, score = agent.search_root(board, 3)
        self.assertIsNone(move)
        self.assertEqual(score, -agent.MATE_SCORE)

    def test_terminal_score_is_ply_adjusted(self) -> None:
        board = chess.Board(CHECKMATED)
        self.assertEqual(agent.negamax(board, 2, ply=4), -agent.MATE_SCORE + 4)

    def test_board_is_unchanged_after_search(self) -> None:
        for fen in (chess.STARTING_FEN, MATE_IN_ONE, HANGING_QUEEN, STALEMATE, CHECKMATED):
            board = chess.Board(fen)
            before = board.fen()
            agent.search_root(board, 3)
            agent.negamax(board, 2)
            self.assertEqual(board.fen(), before)
            self.assertEqual(len(board.move_stack), 0)

    def test_search_is_deterministic(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        first = agent.search_root(chess.Board(fen), 2)
        second = agent.search_root(chess.Board(fen), 2)
        self.assertEqual(first, second)
        first_move = agent.get_move(fen, 60_000)
        agent.reset_game()
        self.assertEqual(first_move, agent.get_move(fen, 60_000))

    def test_evaluation_is_relative_to_side_to_move(self) -> None:
        # White is a queen up. From White's view the score is positive, from Black's negative,
        # and the two are exact negatives of each other at the same material.
        white_to_move = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        black_to_move = chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
        self.assertGreater(agent.negamax(white_to_move, 0), 0)
        self.assertLess(agent.negamax(black_to_move, 0), 0)

    def test_depth_scales_with_the_clock_and_branching(self) -> None:
        # At 0.5 ms a node and 20 root moves: depth 3 is 8000 nodes, 4 s; depth 2 is 0.2 s.
        self.assertEqual(agent.choose_depth(20, 120_000, ms_per_node=0.5), 3)
        self.assertEqual(agent.choose_depth(20, 30_000, ms_per_node=0.5), 2)
        self.assertEqual(agent.choose_depth(20, 3_000, ms_per_node=0.5), 1)
        self.assertEqual(agent.choose_depth(20, 0, ms_per_node=0.5), 1)
        # A wide root drops the depth; a narrow one raises it.
        self.assertEqual(agent.choose_depth(40, 120_000, ms_per_node=0.5), 2)
        self.assertEqual(agent.choose_depth(5, 3_000, ms_per_node=0.5), 3)
        self.assertEqual(agent.choose_depth(0, 120_000, ms_per_node=0.5), 3)
        self.assertGreaterEqual(agent.MS_PER_NODE, agent.MIN_MS_PER_NODE)
        self.assertLessEqual(agent.MS_PER_NODE, agent.MAX_MS_PER_NODE)

    def test_deadline_still_returns_a_move(self) -> None:
        board = chess.Board()
        move, _ = agent.search_root(board, 3, deadline=0.0)
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves)

    def test_root_avoids_repeating_a_seen_position(self) -> None:
        # White is a rook up. Rb1 returns to a position already seen twice this game and
        # is scored as a draw, so the search picks something else.
        start = "4k3/8/8/8/8/8/8/1R2K3 w - - 0 1"
        board = chess.Board(start)
        for uci in ("b1a1", "e8d8", "a1b1", "d8e8"):
            board.push_uci(uci)
        seen = {agent.position_key(chess.Board(start)): 2}
        repeat = chess.Board(board.fen())
        move, score = agent.search_root(repeat, 2, seen=seen)
        self.assertIsNotNone(move)
        assert move is not None
        self.assertGreater(score, 0)
        repeat.push(move)
        self.assertNotIn(agent.position_key(repeat), seen)
        # Without history the same search is free to pick whatever scores best.
        move_without, _ = agent.search_root(chess.Board(board.fen()), 2)
        self.assertIn(move_without, chess.Board(board.fen()).legal_moves)

    def test_promotion_and_en_passant_are_playable(self) -> None:
        # The black king attacks the pawn, so anything but promoting loses it.
        promotion = "8/1P6/k7/8/8/8/8/4K3 w - - 0 1"
        uci = agent.get_move(promotion, 120_000)
        self.assertEqual(uci, "b7b8q")
        en_passant = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"
        board = chess.Board(en_passant)
        uci = agent.get_move(en_passant, 120_000)
        self.assertIn(chess.Move.from_uci(uci), board.legal_moves)


if __name__ == "__main__":
    unittest.main()
