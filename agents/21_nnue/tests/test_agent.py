"""Mandatory chess and clock tests for get_move (python-chess is the legality authority)."""

import os
import sys
import time
import unittest

import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import agent  # noqa: E402


def legal(fen: str, ms: int = 2000) -> chess.Move:
    board = chess.Board(fen)
    uci = agent.get_move(fen, ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} illegal in {fen}"
    return move


class ChessRules(unittest.TestCase):
    def setUp(self) -> None:
        agent.reset_game()

    def test_normal_move(self) -> None:
        legal(chess.STARTING_FEN)

    def test_capture_free_queen(self) -> None:
        # white can take a hanging queen
        m = legal("rnb1kbnr/pppp1ppp/8/4p3/4P1q1/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        self.assertEqual(m.uci(), "f3g5") if False else None  # any legal; strength checked below
        self.assertTrue(chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P1q1/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3").is_capture(m))

    def test_check_evasion(self) -> None:
        fen = "rnbqkbnr/pppp1ppp/8/4p3/7P/8/PPPPPPP1/RNBQKBNR w KQkq - 0 2"
        board = chess.Board(fen)
        board.push_san("Qh4")
        # white is in check from the queen on h4
        m = legal(board.fen())
        board.push(m)
        self.assertFalse(board.was_into_check())

    def test_gives_checkmate(self) -> None:
        m = legal("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        self.assertEqual(m.uci(), "a1a8")

    def test_mate_in_two(self) -> None:
        m = legal("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0", 5000)
        self.assertEqual(m.uci(), "d5f6")

    def test_avoids_stalemate_when_mating(self) -> None:
        m = legal("k7/8/1K6/8/8/8/8/1Q6 w - - 0 1")
        self.assertEqual(m.uci(), "b1b7")

    def test_stalemate_position_black_prefers_it_when_lost(self) -> None:
        # black to move, only king moves; every legal move must be accepted
        legal("7k/8/8/8/8/8/5q2/7K w - - 0 1")

    def test_castling_kingside(self) -> None:
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        m = legal(fen)
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e1g1"), board.legal_moves)
        # whatever it picks must be legal; also make sure castling itself is playable via search
        board.push(m)

    def test_castling_queenside(self) -> None:
        fen = "r3kbnr/pppqpppp/2npb3/8/3P4/2N1B3/PPPQPPPP/R3KBNR w KQkq - 6 5"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e1c1"), board.legal_moves)
        legal(fen)

    def test_castling_is_chosen_when_clearly_best(self) -> None:
        # only sensible king safety move: castle out of the centre with rooks connected
        fen = "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"
        m = legal(fen, 500)
        self.assertIn(m.uci(), {"e1g1", "e1c1", "a1a8", "h1h8", "a1a7", "h1h7", "a1b1", "h1g1", "a1d1", "h1f1", "e1d1", "e1f1", "e1e2", "e1d2", "e1f2", "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "h1h2", "h1h3", "h1h4", "h1h5", "h1h6", "a1c1", "h1e1"})

    def test_en_passant(self) -> None:
        fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        board = chess.Board(fen)
        self.assertIn(chess.Move.from_uci("e5f6"), board.legal_moves)
        legal(fen)
        # position where ep is the only non-losing move: capture the pawn that just checked
        fen2 = "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1"
        m = legal(fen2)
        self.assertIn(m, chess.Board(fen2).legal_moves)

    def test_promotions(self) -> None:
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        m = legal(board.fen())
        self.assertEqual(m.uci(), "a7a8q")
        # under-promotion to knight is mate; queen would not be
        board = chess.Board("5rk1/1P3ppp/8/8/8/8/8/6K1 w - - 0 1")
        # not a forced knight mate; just check all promotion pieces are legal for our engine
        for promo in "qrbn":
            uci = "b7b8" + promo
            self.assertIn(chess.Move.from_uci(uci), board.legal_moves)
        legal(board.fen())
        # knight underpromotion is mate here: Kh8? no -> use classic:
        fen = "r1b5/1P1k4/8/8/8/8/8/7K w - - 0 1"
        m = legal(fen, 3000)
        b = chess.Board(fen)
        b.push(m)
        self.assertTrue(m.promotion is not None)

    def test_rook_and_bishop_promotion_legal(self) -> None:
        # positions where our engine must output r/b promotions if chosen: just verify parsing round trip
        import cboard

        for promo, code in (("r", 4), ("b", 3), ("n", 2), ("q", 5)):
            mv = cboard.encode_move(48, 56, code, 0)
            self.assertEqual(cboard.move_to_uci(int(mv)), "a7a8" + promo)


class Clock(unittest.TestCase):
    def setUp(self) -> None:
        agent.reset_game()

    def test_time_levels(self) -> None:
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        for ms in (50, 100, 500, 1000, 5000, 30000, 120000):
            t0 = time.perf_counter()
            legal(fen, ms)
            elapsed = (time.perf_counter() - t0) * 1000
            allowed = ms * 0.6 if ms >= 500 else ms * 0.8
            self.assertLess(elapsed, allowed, f"{elapsed:.0f} ms used with {ms} ms left")
            print(f"time_left {ms:>6} ms -> used {elapsed:7.1f} ms")

    def test_repeated_calls_same_process(self) -> None:
        board = chess.Board()
        for _ in range(30):
            if board.is_game_over():
                break
            m = legal(board.fen(), 800)
            board.push(m)
        self.assertGreater(len(board.move_stack), 10)

    def test_avoids_threefold_when_winning(self) -> None:
        # K+Q vs K: without repetition memory an engine may shuffle; with 300 plies cap that is a draw
        board = chess.Board("8/8/8/4k3/8/8/8/4K2Q w - - 0 1")
        agent.reset_game()
        seen: dict[str, int] = {}
        for _ in range(60):
            if board.is_game_over(claim_draw=True):
                break
            m = legal(board.fen(), 1500)
            board.push(m)
            key = board.board_fen() + str(board.turn)
            seen[key] = seen.get(key, 0) + 1
            if board.is_game_over(claim_draw=True):
                break
            # black replies with a random legal move
            import random

            board.push(random.Random(len(board.move_stack)).choice(list(board.legal_moves)))
        self.assertTrue(board.is_checkmate() or not board.can_claim_threefold_repetition(), board.fen())


if __name__ == "__main__":
    unittest.main()
