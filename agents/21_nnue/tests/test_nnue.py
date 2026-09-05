"""NNUE plumbing: incremental accumulator == full refresh, colour-flip symmetry, safetensors."""

import os
import random
import sys
import tempfile
import unittest

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cboard as cb  # noqa: E402
import nnue  # noqa: E402


class NNUETest(unittest.TestCase):
    def setUp(self) -> None:
        self.net = nnue.random_net(64, seed=3)
        self.undo = cb.new_undo()
        self.moves = cb.new_movelists()

    def test_incremental_equals_refresh_on_random_games(self) -> None:
        W1, B1, W2, B2 = self.net
        acc = nnue.new_acc(64)
        ref = nnue.new_acc(64)
        rng = random.Random(7)
        for _ in range(30):
            board = chess.Board()
            P = cb.from_board(board)
            nnue.refresh(acc, 0, P, W1, B1)
            ply = 0
            while ply < 100 and not board.is_game_over():
                moves = list(board.legal_moves)
                # prefer "interesting" moves so castling/ep/promotion get exercised
                special = [m for m in moves if board.is_castling(m) or board.is_en_passant(m) or m.promotion]
                mv = rng.choice(special) if special and rng.random() < 0.8 else rng.choice(moves)
                n = cb.gen_moves(P, self.moves[ply], False)
                idx = [i for i in range(n) if cb.move_to_uci(int(self.moves[ply, i])) == mv.uci()]
                self.assertEqual(len(idx), 1, mv.uci())
                self.assertTrue(cb.make_move(P, self.undo, ply, self.moves[ply, idx[0]]))
                nnue.update(acc, ply, P, W1)
                board.push(mv)
                nnue.refresh(ref, ply + 1, P, W1, B1)
                np.testing.assert_array_equal(acc[ply + 1], ref[ply + 1], f"after {mv.uci()} in {board.fen()}")
                ply += 1

    def test_colour_flip_symmetry(self) -> None:
        """Mirroring the board and swapping colours must negate nothing but the perspective:
        eval(pos, stm) == eval(flipped pos, flipped stm)."""
        W1, B1, W2, B2 = self.net
        acc = nnue.new_acc(64)
        for fen in [
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/P4k2/8/8/8/8/1p3K2/8 b - - 0 1",
        ]:
            board = chess.Board(fen)
            P = cb.from_board(board)
            nnue.refresh(acc, 0, P, W1, B1)
            e1 = nnue.evaluate(acc, 0, P[cb.SIDE], W2, B2)
            flipped = board.mirror()  # swaps colours and flips vertically, side to move flips
            Q = cb.from_board(flipped)
            nnue.refresh(acc, 1, Q, W1, B1)
            e2 = nnue.evaluate(acc, 1, Q[cb.SIDE], W2, B2)
            self.assertEqual(int(e1), int(e2), fen)

    def test_safetensors_round_trip(self) -> None:
        W1, B1, W2, B2 = self.net
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "n.safetensors")
            nnue.write_safetensors(path, {"W1": W1, "B1": B1, "W2": W2, "B2": B2}, {"arch": "test"})
            W1b, B1b, W2b, B2b = nnue.load_net(path)
            np.testing.assert_array_equal(W1, W1b)
            np.testing.assert_array_equal(B1, B1b)
            np.testing.assert_array_equal(W2, W2b)
            np.testing.assert_array_equal(B2, B2b)


if __name__ == "__main__":
    unittest.main()
