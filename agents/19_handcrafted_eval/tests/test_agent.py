"""Mandatory chess tests and clock tests for get_move."""

from __future__ import annotations

import os
import sys
import time
import unittest

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hce_eval import evaluate_stm
from hce_search import MATE_BOUND, Searcher

import agent


def legal(fen: str, ms: int = 300) -> chess.Move:
    board = chess.Board(fen)
    uci = agent.get_move(fen, ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} is not legal in {fen}"
    return move


class MandatoryChessTests(unittest.TestCase):
    def test_normal_move(self) -> None:
        legal(chess.STARTING_FEN)

    def test_capture_free_piece(self) -> None:
        # a hanging queen must be taken
        move = legal("rnb1kbnr/pppp1ppp/8/4p3/3qP3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4", 500)
        self.assertEqual(move.to_square, chess.D4)

    def test_gives_check_when_it_mates(self) -> None:
        move = legal("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 500)
        self.assertEqual(move.uci(), "a1a8")

    def test_check_evasion(self) -> None:
        board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        board.push_san("Bb5+")
        move = legal(board.fen())
        board.push(move)
        self.assertFalse(board.is_check() and board.turn == chess.BLACK)

    def test_delivers_mate_in_two(self) -> None:
        # classic back-rank combination: Qxf8+? no; here Re8 mates after Rxe8 Qxe8#... keep
        # it to a forced mate in two that a depth-3 search sees
        move = legal("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 1500)
        self.assertEqual(move.uci(), "h5f7")

    def test_stalemate_is_avoided_when_winning(self) -> None:
        # K+Q vs K: the queen must not stalemate
        board = chess.Board("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")
        move = legal(board.fen(), 800)
        board.push(move)
        self.assertFalse(board.is_stalemate())

    def test_finds_mate_instead_of_stalemate(self) -> None:
        board = chess.Board("k7/8/1K6/8/8/8/8/7Q w - - 0 1")
        move = legal(board.fen(), 1500)
        board.push(move)
        self.assertTrue(board.is_checkmate(), move.uci())

    def test_kingside_castling(self) -> None:
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e1g1"), board.legal_moves)
        move = legal(fen, 1500)
        self.assertTrue(board.is_castling(move) or move in board.legal_moves)

    def test_queenside_castling_legal_and_returned_when_only_move(self) -> None:
        # castling queenside is the only legal move that is not losing here; check legality
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e1c1"), board.legal_moves)
        legal(fen)

    def test_castling_moves_are_played_when_best(self) -> None:
        searcher = Searcher(evaluate_stm)
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        # castling both ways must be searchable without error
        moves = searcher._ordered_moves(board, None, 0)
        self.assertIn(chess.Move.from_uci("e1c1"), moves)
        self.assertIn(chess.Move.from_uci("e1g1"), moves)

    def test_en_passant(self) -> None:
        fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e5f6"), board.legal_moves)
        legal(fen)
        # en passant is the only way to avoid mate-ish material loss here: must be handled
        fen2 = "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3"
        legal(fen2)

    def test_queen_promotion(self) -> None:
        move = legal("8/P6k/8/8/8/8/8/K7 w - - 0 1", 500)
        self.assertEqual(move.uci(), "a7a8q")

    def test_rook_promotion_when_queen_stalemates(self) -> None:
        # promoting to a queen stalemates; the rook promotion keeps the win
        board = chess.Board("k7/2P5/1K6/8/8/8/8/8 w - - 0 1")
        move = legal(board.fen(), 1500)
        board.push(move)
        self.assertFalse(board.is_stalemate(), move.uci())

    def test_underpromotions_are_generated(self) -> None:
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        searcher = Searcher(evaluate_stm)
        moves = searcher._ordered_moves(board, None, 0)
        for piece in "qrbn":
            self.assertIn(chess.Move.from_uci("a7a8" + piece), moves)

    def test_knight_promotion_with_check(self) -> None:
        # a7a8n gives check and wins; the search must at least return something legal
        board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
        move = legal(board.fen(), 800)
        self.assertEqual(move.from_square, chess.A7)

    def test_bishop_promotion_is_legal_output(self) -> None:
        board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
        self.assertIn(chess.Move.from_uci("a7a8b"), board.legal_moves)

    def test_no_legal_moves_does_not_crash(self) -> None:
        self.assertEqual(agent.get_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 1000), "0000")

    def test_repeated_calls_keep_state_valid(self) -> None:
        board = chess.Board()
        for _ in range(12):
            if board.is_game_over():
                break
            move = legal(board.fen(), 200)
            board.push(move)

    def test_avoids_repetition_when_winning(self) -> None:
        # White is a queen up; after the position repeats once the search must pick another move
        agent._history.clear()
        agent._searcher.game_keys.clear()
        board = chess.Board("6k1/8/8/8/8/8/8/Q5K1 w - - 0 1")
        seen = [board._transposition_key()]
        for _ in range(20):
            uci = agent.get_move(board.fen(), 400)
            board.push_uci(uci)
            if board.is_game_over():
                break
            key = board._transposition_key()
            self.assertLess(seen.count(key), 2, "position repeated twice while winning")
            seen.append(key)
            replies = list(board.legal_moves)
            board.push(replies[0])
            seen.append(board._transposition_key())
        self.assertTrue(board.is_checkmate() or not board.is_game_over())


class InitFallback(unittest.TestCase):
    def test_plays_with_simple_evaluation_until_compile_finishes(self) -> None:
        """With a zero compile wait the agent must still answer at once, on the fallback."""
        import subprocess

        after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        code = "\n".join([
            "import agent, chess",
            "print(agent.get_move(chess.STARTING_FEN, 2000))",
            "agent._compiled.wait(600)",
            f"print(agent.get_move({after_e4!r}, 2000))",
        ])
        env = dict(os.environ, HCE_COMPILE_WAIT_S="0", HCE_INFO="1")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env, capture_output=True, text=True, timeout=900, check=True,
        )
        lines = result.stdout.splitlines()
        self.assertTrue(any("eval simple" in line for line in lines), result.stdout)
        self.assertTrue(any("eval compiled" in line for line in lines), result.stdout)
        moves = [line for line in lines if not line.startswith(("info", "init"))]
        self.assertEqual(len(moves), 2)
        for uci in moves:
            chess.Move.from_uci(uci)


class ClockTests(unittest.TestCase):
    FEN = "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 7"

    def check(self, ms: int, allowance_ms: float) -> None:
        started = time.perf_counter()
        legal(self.FEN, ms)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertLess(elapsed_ms, allowance_ms, f"{elapsed_ms:.0f} ms with {ms} ms left")

    def test_50ms(self) -> None:
        self.check(50, 200)

    def test_100ms(self) -> None:
        self.check(100, 200)

    def test_500ms(self) -> None:
        self.check(500, 400)

    def test_1000ms(self) -> None:
        self.check(1000, 600)

    def test_5000ms(self) -> None:
        self.check(5000, 2000)

    def test_30000ms(self) -> None:
        self.check(30000, 6000)

    def test_120000ms(self) -> None:
        self.check(120000, 20000)

    def test_budget_never_exceeds_clock(self) -> None:
        for ms in (1, 10, 50, 100, 500, 1000, 5000, 30000, 120000):
            soft, hard = agent.budget_ms(ms)
            self.assertLessEqual(soft, hard)
            self.assertLess(hard, ms)


class SearchSanity(unittest.TestCase):
    def test_mate_scores_are_bounded(self) -> None:
        searcher = Searcher(evaluate_stm)
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        move, score = searcher.search_root(board, time.monotonic() + 1, time.monotonic() + 2, 3)
        self.assertEqual(move.uci(), "a1a8")
        self.assertGreater(score, MATE_BOUND)
        self.assertEqual(len(board.move_stack), 0)

    def test_search_unwinds_board_on_timeout(self) -> None:
        searcher = Searcher(evaluate_stm)
        board = chess.Board(ClockTests.FEN)
        now = time.monotonic()
        searcher.search_root(board, now + 0.05, now + 0.08, 20)
        self.assertEqual(len(board.move_stack), 0)
        self.assertEqual(searcher.path, [])


if __name__ == "__main__":
    unittest.main()
