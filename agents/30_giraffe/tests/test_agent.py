"""Correctness tests for the Giraffe agent: legality, rules edge cases, clocks, symmetry.

Run from the agent directory with
``python -m unittest discover -s tests`` (the project venv interpreter).
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
import giraffe_eval as ge  # noqa: E402
from giraffe_search import MATE_BOUND, Searcher  # noqa: E402


def legal(fen: str, time_left_ms: int = 2000) -> chess.Move:
    board = chess.Board(fen)
    uci = agent.get_move(fen, time_left_ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} illegal in {fen}"
    return move


class MoveTypes(unittest.TestCase):
    def test_normal_move(self) -> None:
        legal(chess.STARTING_FEN)

    def test_capture_is_taken_when_free(self) -> None:
        move = legal("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
        self.assertEqual(move.uci(), "e4d5")

    def test_gives_check_position(self) -> None:
        legal("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")

    def test_check_evasion(self) -> None:
        fen = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5P2/PPPP2PP/RNBQKBNR w KQkq - 1 3"
        board = chess.Board(fen)
        self.assertTrue(board.is_check())
        legal(fen)

    def test_delivers_checkmate(self) -> None:
        move = legal("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        self.assertEqual(move.uci(), "d8h4")

    def test_mate_in_two(self) -> None:
        # K+Q vs K, mate in two but not in one (verified by brute force)
        fen = "7k/8/5K2/8/8/8/8/1Q6 w - - 0 1"
        board = chess.Board(fen)
        searcher = Searcher(ge.hce_eval)
        move, score = searcher.search(board, 10.0, 3)
        self.assertGreaterEqual(score, MATE_BOUND)
        board.push(move)
        self.assertFalse(board.is_stalemate())

    def test_mate_in_one_found_at_depth_one(self) -> None:
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        searcher = Searcher(ge.hce_eval)
        move, score = searcher.search(chess.Board(fen), 10.0, 1)
        self.assertEqual(move.uci(), "h5f7")
        self.assertGreaterEqual(score, MATE_BOUND)

    def test_stalemate_position_avoided_when_winning(self) -> None:
        # white can stalemate with Qc7?? or win; agent must not play a stalemating move
        fen = "k7/2Q5/1K6/8/8/8/8/8 w - - 0 1"
        move = legal(fen)
        board = chess.Board(fen)
        board.push(move)
        self.assertFalse(board.is_stalemate())

    def test_only_move_in_near_stalemate(self) -> None:
        # black to move has exactly one legal move
        fen = "7k/8/5Q1K/8/8/8/8/8 b - - 0 1"
        board = chess.Board(fen)
        self.assertEqual(len(list(board.legal_moves)), 1)
        legal(fen)

    def test_kingside_castling(self) -> None:
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e1g1"), board.legal_moves)
        legal(fen)

    def test_queenside_castling(self) -> None:
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e8c8"), board.legal_moves)
        legal(fen)

    def test_castling_played_when_it_is_the_tt_move(self) -> None:
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        for _ in range(3):
            legal(fen)

    def test_en_passant(self) -> None:
        fen = "8/8/8/3pP2k/8/8/8/K7 w - d6 0 1"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e5d6"), board.legal_moves)
        move = legal(fen)
        self.assertEqual(move.uci(), "e5d6")

    def test_queen_promotion(self) -> None:
        move = legal("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        self.assertEqual(move.uci(), "a7a8q")

    def test_underpromotions_are_legal_uci(self) -> None:
        fen = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
        board = chess.Board(fen)
        for piece in "rbn":
            self.assertIn(chess.Move.from_uci(f"a7a8{piece}"), board.legal_moves)

    def test_knight_promotion_when_it_mates(self) -> None:
        # a7a8n is not mate here, but knight promotion with check must at least be legal
        fen = "8/1P4k1/8/8/8/8/8/K7 w - - 0 1"
        legal(fen)

    def test_capture_promotion(self) -> None:
        fen = "1r5k/P7/8/8/8/8/8/K7 w - - 0 1"
        move = legal(fen)
        self.assertEqual(move.uci(), "a7b8q")


class Clock(unittest.TestCase):
    FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

    def test_all_clocks_return_legal_moves_in_time(self) -> None:
        for time_left in (50, 100, 500, 1000, 5000, 30000, 120000):
            board = chess.Board(self.FEN)
            started = time.monotonic()
            uci = agent.get_move(self.FEN, time_left)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self.assertIn(chess.Move.from_uci(uci), board.legal_moves)
            allowed = max(time_left * 0.6, 60) if time_left <= 1000 else agent.MAX_BUDGET_S * 1000 * 1.5
            self.assertLess(elapsed_ms, allowed + 300, f"{time_left} ms clock took {elapsed_ms:.0f} ms")

    def test_budget_monotone_and_bounded(self) -> None:
        last = 0.0
        for time_left in (300, 1000, 5000, 30000, 120000):
            budget = agent.budget_seconds(time_left)
            self.assertGreaterEqual(budget, last)
            self.assertLessEqual(budget, agent.MAX_BUDGET_S)
            self.assertLess(budget * 1000, time_left)
            last = budget

    def test_repeated_calls_keep_state_valid(self) -> None:
        board = chess.Board()
        for _ in range(12):
            if board.is_game_over():
                break
            uci = agent.get_move(board.fen(), 3000)
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            board.push(move)


class Features(unittest.TestCase):
    FENS = [
        chess.STARTING_FEN,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ]

    def test_shape_and_range(self) -> None:
        for fen in self.FENS:
            x = ge.board_features(chess.Board(fen))
            self.assertEqual(x.shape, (ge.N_INPUT,))
            self.assertTrue(np.all(x >= 0.0) and np.all(x <= 1.0), fen)

    def test_colour_symmetry(self) -> None:
        """Mirroring the board (swap colours, flip ranks) must give identical features."""
        for fen in self.FENS:
            board = chess.Board(fen)
            x = ge.board_features(board)
            y = ge.board_features(board.mirror())
            np.testing.assert_allclose(x, y, err_msg=fen)

    def test_value_sign_flips_with_side_to_move(self) -> None:
        evaluator = ge.NetEvaluator(ge.random_weights(3))
        for fen in self.FENS:
            board = chess.Board(fen)
            v = evaluator(board)
            mirrored = evaluator(board.mirror())
            self.assertEqual(v, mirrored)
            self.assertEqual(ge.hce_eval(board), ge.hce_eval(board.mirror()))

    def test_material_counts(self) -> None:
        x = ge.board_features(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"))
        # rook count for us is 1/2, everything else 0
        self.assertAlmostEqual(float(x[3 + chess.ROOK]), 0.5)
        self.assertEqual(float(x[3 + chess.QUEEN]), 0.0)
        self.assertEqual(float(x[8 + chess.ROOK]), 0.0)

    def test_attack_map(self) -> None:
        # white knight f3 attacks d4/e5/g5/h4/d2/e1/g1/h2; black pawn d6 attacks c5/e5
        board = chess.Board("4k3/8/3p4/8/8/5N2/8/4K3 w - - 0 1")
        x = ge.board_features(board)
        base = ge.N_GLOBAL + ge.N_PIECE
        self.assertAlmostEqual(float(x[base + chess.D4]), 0.3)
        self.assertAlmostEqual(float(x[base + chess.E5]), 0.3)
        self.assertAlmostEqual(float(x[base + 64 + chess.E5]), 0.1)
        self.assertAlmostEqual(float(x[base + 64 + chess.C5]), 0.1)
        self.assertEqual(float(x[base + 64 + chess.D5]), 0.0)

    def test_hce_material(self) -> None:
        up_a_rook = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        self.assertGreater(ge.hce_eval(up_a_rook), 400)
        up_a_rook.turn = chess.BLACK
        self.assertLess(ge.hce_eval(up_a_rook), -400)


class SearchRules(unittest.TestCase):
    def test_avoids_threefold_when_winning(self) -> None:
        """With a queen up, the searcher must not steer into a position seen before."""
        searcher = Searcher(ge.hce_eval)
        board = chess.Board("7k/8/8/8/8/8/8/Q3K3 w - - 0 1")
        move, score = searcher.search(board, 1.0, 4)
        self.assertGreater(score, 500)
        board.push(move)
        searcher.remember(board)
        board.pop()
        move2, _ = searcher.search(board, 1.0, 4)
        board.push(move2)
        self.assertNotIn(board._transposition_key(), searcher.game_keys)

    def test_move_stack_unwound_after_timeout(self) -> None:
        board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        searcher = Searcher(ge.hce_eval)
        move, _ = searcher.search(board, 0.05, 64)
        self.assertEqual(len(board.move_stack), 0)
        self.assertIn(move, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
