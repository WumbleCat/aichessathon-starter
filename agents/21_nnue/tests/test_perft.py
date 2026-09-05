"""Perft: our movegen/make/unmake must count exactly like python-chess."""

import os
import sys
import time
import unittest

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cboard

POSITIONS = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890]),
    # black to move / ep / promotions
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", [43, 2002]),
    ("8/8/8/8/k2Pp3/8/8/4K3 b - d3 0 1", [7, 47]),
    ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", [24, 496, 9483]),
]


def python_chess_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += python_chess_perft(board, depth - 1)
        board.pop()
    return total


class PerftTest(unittest.TestCase):
    def test_perft_matches(self) -> None:
        undo = cboard.new_undo()
        moves = cboard.new_movelists()
        for fen, counts in POSITIONS:
            P = cboard.from_board(chess.Board(fen))
            h0 = int(P[cboard.HASH])
            for depth, expected in enumerate(counts, start=1):
                got = cboard.perft(P, undo, moves, 0, depth)
                if expected < 100_000:
                    expected = python_chess_perft(chess.Board(fen), depth)
                self.assertEqual(got, expected, f"{fen} depth {depth}")
            self.assertEqual(int(P[cboard.HASH]), h0)
            self.assertEqual(int(P[cboard.HASH]), int(cboard.compute_hash(P)))

    def test_random_games_hash_and_moves(self) -> None:
        """Walk random games with python-chess and check move sets and hash consistency."""
        rng = np.random.default_rng(1)
        undo = cboard.new_undo()
        moves = cboard.new_movelists()
        for _ in range(12):
            board = chess.Board()
            P = cboard.from_board(board)
            for _ply in range(120):
                legal = {m.uci() for m in board.legal_moves}
                n = cboard.gen_moves(P, moves[0], cboard.ALL_MOVES)
                ours = set()
                for i in range(n):
                    if cboard.make_move(P, undo, 0, moves[0, i]):
                        ours.add(cboard.move_to_uci(int(moves[0, i])))
                    cboard.unmake_move(P, undo, 0)
                self.assertEqual(ours, legal, board.fen())
                if not legal:
                    break
                mv = rng.choice(sorted(legal))
                board.push_uci(mv)
                idx = -1
                for i in range(n):
                    if cboard.move_to_uci(int(moves[0, i])) == mv:
                        idx = i
                self.assertTrue(cboard.make_move(P, undo, 0, moves[0, idx]))
                # after making the move, P must equal a fresh parse of the new position
                Q = cboard.from_board(board)
                for k in range(cboard.LAST_PIECE):
                    self.assertEqual(int(P[k]), int(Q[k]), f"slot {k} after {mv} in {board.fen()}")
                self.assertEqual(int(P[cboard.HASH]), int(cboard.compute_hash(P)))
                self.assertEqual(int(P[cboard.NONPAWN]), int(Q[cboard.NONPAWN]))
                if board.is_game_over():
                    break

    def test_speed(self) -> None:
        undo = cboard.new_undo()
        moves = cboard.new_movelists()
        P = cboard.from_board(
            chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        )
        cboard.perft(P, undo, moves, 0, 2)
        t0 = time.perf_counter()
        n = cboard.perft(P, undo, moves, 0, 4)
        dt = time.perf_counter() - t0
        print(f"perft 4 kiwipete: {n} nodes in {dt:.2f}s = {n / dt / 1e6:.2f} Mnps")


if __name__ == "__main__":
    unittest.main()
