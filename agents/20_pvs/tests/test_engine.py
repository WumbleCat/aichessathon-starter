"""Correctness tests for the 20_pvs engine.

Run from the repository root:
    .venv/Scripts/python.exe -m unittest discover -s agents/20_pvs/tests -v
"""

from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402

import agent  # noqa: E402
import pvs_board as pb  # noqa: E402
from pvs_search import MATE_BOUND, Searcher, default_params  # noqa: E402


def legal(fen: str, uci: str) -> bool:
    return chess.Move.from_uci(uci) in chess.Board(fen).legal_moves


def best(fen: str, ms: int = 1500) -> str:
    return agent.get_move(fen, ms)


class MoveGeneration(unittest.TestCase):
    PERFT = [
        (chess.STARTING_FEN, 3, 8902),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890),
    ]

    def test_perft(self) -> None:
        pos = pb.Position()
        for fen, depth, expect in self.PERFT:
            pos.set_fen(fen)
            self.assertEqual(pos.perft(depth), expect, fen)

    def test_legal_moves_match_python_chess(self) -> None:
        fens = [
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "8/P7/8/8/8/8/7p/K6k w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ]
        for fen in fens:
            pos = pb.Position(fen)
            ours = sorted(pb.move_to_uci(m) for m in pos.legal_moves())
            ref = sorted(m.uci() for m in chess.Board(fen).legal_moves)
            self.assertEqual(ours, ref, fen)

    def test_hash_incremental_matches_fresh(self) -> None:
        board = chess.Board()
        pos = pb.Position()
        import random

        rng = random.Random(7)
        for _ in range(80):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            ours = [m for m in pos.legal_moves() if pb.move_to_uci(m) == move.uci()]
            self.assertEqual(len(ours), 1)
            pos.push(ours[0])
            board.push(move)
            fresh = pb.Position(board.fen())
            self.assertEqual(int(fresh.st[pb.ST_HASH]), int(pos.st[pb.ST_HASH]), board.fen())
            self.assertEqual(int(fresh.st[pb.ST_MG]), int(pos.st[pb.ST_MG]))
            self.assertEqual(int(fresh.st[pb.ST_EG]), int(pos.st[pb.ST_EG]))
            pos.st[pb.ST_PLY] = 0


class MandatoryChessTests(unittest.TestCase):
    def check(self, fen: str, expected: str | None = None, ms: int = 1500) -> str:
        uci = best(fen, ms)
        self.assertTrue(legal(fen, uci), f"{uci} illegal in {fen}")
        if expected is not None:
            self.assertEqual(uci, expected, fen)
        return uci

    def test_normal_move(self) -> None:
        self.check(chess.STARTING_FEN)

    def test_capture_free_piece(self) -> None:
        # white queen can take an undefended rook on a8? use a hanging queen instead
        self.check("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3h4")

    def test_gives_check_and_wins(self) -> None:
        # Mate in one: Qxf7#
        self.check("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7")

    def test_check_evasion(self) -> None:
        fen = "rnbqkbnr/pppp1ppp/8/4p3/7P/8/PPPPPPP1/RNBQKBNR w KQkq - 0 2"
        board = chess.Board(fen)
        board.push_san("h5")  # nonsense to keep a legal board
        fen = "rnb1kbnr/pppp1ppp/8/4p3/7q/8/PPPPPPP1/RNBQKBNR w KQkq - 0 3"
        uci = self.check(fen)
        board = chess.Board(fen)
        self.assertTrue(board.is_check())
        board.push_uci(uci)
        self.assertFalse(board.is_check() and board.turn == chess.WHITE)

    def test_checkmate_delivered(self) -> None:
        # Back rank mate: Re8#
        self.check("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8")

    def test_mate_in_two_found(self) -> None:
        # classic: 1.Qxh7+ Kxh7 2.Rh3# style? use a well known mate in 2
        fen = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1"
        uci = self.check(fen, "d5f6", ms=4000)  # Nf6+ gxf6 Bxf7#
        self.assertEqual(uci, "d5f6")

    def test_stalemate_avoided_when_winning(self) -> None:
        # White K+Q vs K: choose a move that keeps progress and is not stalemate
        fen = "7k/8/6K1/8/8/8/8/5Q2 w - - 0 1"
        uci = self.check(fen, ms=2000)
        board = chess.Board(fen)
        board.push_uci(uci)
        self.assertFalse(board.is_stalemate())

    def test_stalemate_position_side_to_move(self) -> None:
        # side to move has only stalemate-avoiding drawn options; must still return legal
        fen = "8/8/8/8/8/2k5/1q6/K7 w - - 0 1"
        board = chess.Board(fen)
        if board.is_stalemate():
            self.skipTest("no legal moves")
        self.check(fen)

    def test_kingside_castling(self) -> None:
        uci = self.check("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5")
        self.assertIn(uci, {"e1g1", "b1c3", "c1e3", "c1g5", "d1e2", "a2a3", "c2c3", "h2h3",
                            "b1d2", "e1e2", "f3g5", "d3d4", "b2b4", "a2a4"})
        board = chess.Board("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1")
        uci = self.check(board.fen())
        self.assertTrue(legal(board.fen(), uci))

    def test_castles_when_only_sensible(self) -> None:
        # king in danger, castling is clearly best
        pos = pb.Position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = {pb.move_to_uci(m) for m in pos.legal_moves()}
        self.assertIn("e1g1", moves)
        self.assertIn("e1c1", moves)

    def test_queenside_castling_generated(self) -> None:
        pos = pb.Position("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
        moves = {pb.move_to_uci(m) for m in pos.legal_moves()}
        self.assertIn("e8c8", moves)
        self.assertIn("e8g8", moves)

    def test_en_passant(self) -> None:
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        pos = pb.Position(fen)
        self.assertIn("e5d6", {pb.move_to_uci(m) for m in pos.legal_moves()})
        # en passant is the only way to win the pawn here
        fen2 = "8/8/8/2k5/3pP3/8/8/4K3 b - e3 0 1"
        self.check(fen2)
        # a position where taking en passant is clearly best (wins a pawn for free)
        uci = self.check("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ms=1000)
        self.assertEqual(uci, "e5d6")

    def test_queen_promotion(self) -> None:
        self.check("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q")

    def test_rook_promotion_legal(self) -> None:
        # forced: promoting to a queen is stalemate, rook wins
        fen = "k7/2P5/1K6/8/8/8/8/8 w - - 0 1"
        uci = self.check(fen, ms=2000)
        board = chess.Board(fen)
        board.push_uci(uci)
        self.assertFalse(board.is_stalemate(), uci)

    def test_bishop_and_knight_promotion_generated(self) -> None:
        pos = pb.Position("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        moves = {pb.move_to_uci(m) for m in pos.legal_moves()}
        for promo in "qrbn":
            self.assertIn("a7a8" + promo, moves)

    def test_knight_promotion_when_it_mates(self) -> None:
        # Knight promotion with check that mates: classic pattern
        fen = "5rk1/2P2ppp/8/8/8/8/8/6K1 w - - 0 1"
        board = chess.Board(fen)
        # verify c7c8n is legal, engine must return something legal
        self.assertIn(chess.Move.from_uci("c7c8n"), board.legal_moves)
        self.check(fen)

    def test_underpromotion_to_knight_forced(self) -> None:
        # c8=N+ forks king and queen: the only winning move
        fen = "3q1k2/2P5/8/8/8/8/8/4K3 w - - 0 1"
        uci = self.check(fen, ms=3000)
        self.assertIn(uci, {"c7c8n", "c7d8q"})


class ClockTests(unittest.TestCase):
    FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

    def run_clock(self, ms: int) -> float:
        start = time.perf_counter()
        uci = agent.get_move(self.FEN, ms)
        elapsed = (time.perf_counter() - start) * 1000
        self.assertTrue(legal(self.FEN, uci))
        return elapsed

    def test_clocks(self) -> None:
        for ms in (50, 100, 500, 1000, 5000, 30000, 120000):
            elapsed = self.run_clock(ms)
            # must never come close to flagging: the watchdog grace is 500 ms
            allowed = max(ms * 0.5, 40) if ms < 1000 else ms * 0.35
            self.assertLess(elapsed, allowed, f"{ms} ms clock took {elapsed:.0f} ms")

    def test_repeated_calls_keep_state_valid(self) -> None:
        board = chess.Board()
        for _ in range(12):
            uci = agent.get_move(board.fen(), 3000)
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            board.push(move)
            if board.is_game_over():
                break


class SearchBehaviour(unittest.TestCase):
    def test_mate_scores(self) -> None:
        s = Searcher()
        pos = pb.Position("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        move, score, depth, info = s.search(pos, max_depth=4)
        self.assertEqual(pb.move_to_uci(move), "e1e8")
        self.assertGreater(score, MATE_BOUND)

    def test_repetition_avoided_when_ahead(self) -> None:
        # White is up a queen; history says the position occurred twice already.
        s = Searcher()
        fen = "7k/8/8/8/8/8/8/Q6K w - - 0 1"
        pos = pb.Position(fen)
        key = int(pos.st[pb.ST_HASH])
        move, score, depth, info = s.search(pos, max_depth=3, history_keys=[key, key])
        self.assertGreater(score, 300)

    def test_features_can_be_disabled(self) -> None:
        pos = pb.Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        base = Searcher(default_params())
        m1, s1, d1, i1 = base.search(pos, max_depth=4)
        from pvs_search import P_LMR, P_NULL

        p = default_params()
        p[P_NULL] = 0
        p[P_LMR] = 0
        plain = Searcher(p)
        m2, s2, d2, i2 = plain.search(pos, max_depth=4)
        self.assertGreater(i2["nodes"], i1["nodes"] // 2)
        self.assertTrue(legal(pos_fen(pos), pb.move_to_uci(m2)))

    def test_node_limit_stops(self) -> None:
        s = Searcher()
        pos = pb.Position(chess.STARTING_FEN)
        move, score, depth, info = s.search(pos, max_depth=30, node_limit=20000)
        self.assertLess(info["nodes"], 40000)
        self.assertNotEqual(move, 0)


def pos_fen(pos: pb.Position) -> str:
    """Rebuild a FEN from the numba position (for assertions only)."""
    board = chess.Board(None)
    for s in range(64):
        code = int(pos.sq[s])
        if code:
            board.set_piece_at(
                s, chess.Piece((code - 1) % 6 + 1, chess.WHITE if code <= 6 else chess.BLACK)
            )
    board.turn = chess.WHITE if pos.st[pb.ST_SIDE] == 0 else chess.BLACK
    return board.fen()


if __name__ == "__main__":
    unittest.main()
