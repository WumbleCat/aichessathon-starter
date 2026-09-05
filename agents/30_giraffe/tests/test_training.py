"""The torch training model and the numba inference network must agree exactly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "training"))

import giraffe_eval as ge  # noqa: E402
from model import GiraffeNet, cp_to_target  # noqa: E402


class TorchNumbaParity(unittest.TestCase):
    def test_round_trip_and_forward(self) -> None:
        torch.manual_seed(0)
        model = GiraffeNet()
        flat = model.to_flat()
        self.assertEqual(flat.shape, (ge.N_WEIGHTS,))
        clone = GiraffeNet()
        clone.load_flat(flat)
        np.testing.assert_array_equal(clone.to_flat(), flat)

        evaluator = ge.NetEvaluator(flat)
        fens = [
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        ]
        xs = np.stack([ge.board_features(chess.Board(f)) for f in fens])
        with torch.no_grad():
            torch_cp = model(torch.from_numpy(xs)).numpy() * ge.OUT_SCALE
        for fen, expected in zip(fens, torch_cp, strict=True):
            board = chess.Board(fen)
            total = ge.net_eval_bb(*ge._bitboards(board), evaluator.weights, evaluator.scratch)
            residual = float(total) - ge.hce_eval(board)  # the network only predicts the residual
            self.assertAlmostEqual(residual, float(expected), delta=0.5, msg=fen)
            self.assertAlmostEqual(evaluator.residual(board), float(expected), delta=0.5, msg=fen)

    def test_target_transform(self) -> None:
        cp = np.array([0.0, 100.0, -100.0, 5000.0], dtype=np.float32)
        t = cp_to_target(cp)
        self.assertEqual(t[0], 0.0)
        self.assertAlmostEqual(float(t[1]), -float(t[2]))
        self.assertLess(float(t[3]), 1.0)


if __name__ == "__main__":
    unittest.main()
