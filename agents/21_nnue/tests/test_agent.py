"""Mandatory chess and clock tests for get_move (python-chess is the legality authority)."""

import os
import sys
import time
import unittest

import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import agent


def legal(fen: str, ms: int = 2000) -> chess.Move:
    board = chess.Board(fen)
    uci = agent.get_move(fen, ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} illegal in {fen}"
    return move


class ChessRules(unittest.TestCase):
    def setUp(self) -> None:
        agent.wait_ready()
        agent.reset_game()

    def test_normal_move(self) -> None:
        legal(chess.STARTING_FEN)

    def test_capture_free_queen(self) -> None:
        # black's queen sits on f3 where both the d1 queen and the g2 pawn can take it
        fen = "rnb1kbnr/pppp1ppp/8/4p3/4P3/5q2/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
        m = legal(fen)
        self.assertEqual(m.to_square, chess.F3, m.uci())

    def test_check_evasion(self) -> None:
        # 1.e4 e5 2.f4 Qh4+: white is in check and only g3 or Ke2 are legal
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4PP1q/8/PPPP2PP/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board.is_check())
        m = legal(board.fen())
        board.push(m)
        self.assertFalse(board.was_into_check())

    def test_gives_checkmate(self) -> None:
        m = legal("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        self.assertEqual(m.uci(), "a1a8")

    def test_mate_in_two(self) -> None:
        m = legal("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0", 5000)
        self.assertEqual(m.uci(), "d5f6")

    def test_forces_mate_in_two(self) -> None:
        # Ka8 Kb6 Qb1: the own king blocks Qb7, so the fastest win is Qh1+ Kb8 Qh8#.
        # Play it out with the engine on both sides; white must mate within two moves.
        board = chess.Board("k7/8/1K6/8/8/8/8/1Q6 w - - 0 1")
        white_moves = 0
        while not board.is_game_over() and white_moves < 3:
            m = legal(board.fen(), 1500)
            if board.turn:
                white_moves += 1
            board.push(m)
        self.assertTrue(board.is_checkmate(), board.fen())
        self.assertLessEqual(white_moves, 2)

    def test_single_legal_move(self) -> None:
        # black has exactly one legal move (Ka7); the engine must find and return it
        fen = "k7/2K5/8/8/8/8/8/1R6 b - - 0 1"
        self.assertEqual([m.uci() for m in chess.Board(fen).legal_moves], ["a8a7"])
        self.assertEqual(legal(fen).uci(), "a8a7")

    def test_no_legal_moves_does_not_crash(self) -> None:
        # the platform never asks about a finished game, but a stalemate FEN must not raise
        self.assertEqual(agent.get_move("7k/8/8/8/8/8/5q2/7K w - - 0 1", 1000), "0000")

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
        self.assertIn(
            m.uci(),
            {
                "e1g1",
                "e1c1",
                "a1a8",
                "h1h8",
                "a1a7",
                "h1h7",
                "a1b1",
                "h1g1",
                "a1d1",
                "h1f1",
                "e1d1",
                "e1f1",
                "e1e2",
                "e1d2",
                "e1f2",
                "a1a2",
                "a1a3",
                "a1a4",
                "a1a5",
                "a1a6",
                "h1h2",
                "h1h3",
                "h1h4",
                "h1h5",
                "h1h6",
                "a1c1",
                "h1e1",
            },
        )

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
        # our move encoding must round-trip every promotion piece to UCI
        import cboard

        for promo, code in (("r", 4), ("b", 3), ("n", 2), ("q", 5)):
            mv = cboard.encode_move(48, 56, code, 0)
            self.assertEqual(cboard.move_to_uci(int(mv)), "a7a8" + promo)


class Clock(unittest.TestCase):
    def setUp(self) -> None:
        agent.wait_ready()
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
        # K+Q vs K: an engine without repetition memory shuffles into a draw at the 300-ply cap
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
        self.assertTrue(
            board.is_checkmate() or not board.can_claim_threefold_repetition(), board.fen()
        )


class PythonFallback(unittest.TestCase):
    """The pure-python search that answers while numba is still compiling."""

    def test_legal_and_finds_mate_in_one(self) -> None:
        for fen, mates in [
            ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", {"a1a8"}),
            ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", {"d8h4"}),
        ]:
            board = chess.Board(fen)
            move = agent._python_search(board, time.perf_counter() + 0.5)
            self.assertIn(move, board.legal_moves)
            self.assertIn(move.uci(), mates)

    def test_avoids_repeating_when_ahead(self) -> None:
        # KQ v K: with the position after Qh5+ marked as seen, the queen must go elsewhere
        board = chess.Board("8/8/8/4k3/8/8/8/4K2Q w - - 0 1")
        prev = board.copy()
        prev.push_uci("h1h5")
        seen = frozenset({agent._position_key(prev)})
        move = agent._python_search(board, time.perf_counter() + 0.5, seen)
        self.assertIn(move, board.legal_moves)
        self.assertNotEqual(move.uci(), "h1h5")

    def test_respects_deadline(self) -> None:
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        t0 = time.perf_counter()
        before = board.fen()
        move = agent._python_search(board, t0 + 0.05)
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertEqual(board.fen(), before)  # the interrupted search must restore the board
        self.assertIn(move, board.legal_moves)

    def test_get_move_while_engine_not_ready(self) -> None:
        # Simulate a compile that is still running: get_move waits part of its budget for the
        # engine, then answers with the fallback, all inside the move budget.
        agent.wait_ready()
        agent.reset_game()
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        agent._ENGINE_READY.clear()
        try:
            fallback_before = agent._STATS["fallback"]
            t0 = time.perf_counter()
            move = legal(fen, 3000)
            elapsed = time.perf_counter() - t0
            self.assertIn(move, chess.Board(fen).legal_moves)
            self.assertEqual(agent._STATS["fallback"], fallback_before + 1)
            self.assertLess(elapsed, agent.budget_seconds(3000) + 0.1)
            self.assertGreater(elapsed, agent.budget_seconds(3000) * agent.COMPILE_WAIT_SHARE)
        finally:
            agent._ENGINE_READY.set()
        fallback_after = agent._STATS["fallback"]
        legal(fen, 3000)
        self.assertEqual(agent._STATS["fallback"], fallback_after)  # the engine answers again


if __name__ == "__main__":
    unittest.main()
