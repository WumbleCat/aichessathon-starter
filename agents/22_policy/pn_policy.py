"""Batch-1 CPU inference for the policy network.

Two back ends, same weights:
  * ``NumpyPolicy``: pure numpy (im2col + matmul), no torch import at game time.
  * ``OnnxPolicy``: onnxruntime session on ``policy.onnx`` when it is present and faster.

``load_policy`` picks whichever is available and returns an object with
``prior(board) -> dict[chess.Move, float]``: a softmax over the LEGAL moves only.
"""

from __future__ import annotations

import math
import os
import time

import chess
import numpy as np
from pn_encoding import NUM_MOVE_PLANES, encode_board, move_to_index


class NumpyPolicy:
    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.channels = int(data["meta"][0])
        self.blocks = int(data["meta"][1])
        self.w = {k: data[k] for k in data.files}
        self.num_params = sum(int(v.size) for k, v in self.w.items() if k != "meta")
        # (2*C, 8, 8)? no: pad buffers are allocated per call, cheap for 8x8
        self._pad = np.zeros((self.channels, 10, 10), dtype=np.float32)
        self._pad_in = np.zeros((18, 10, 10), dtype=np.float32)

    @staticmethod
    def _conv3(x: np.ndarray, pad: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """x: (Cin, 8, 8) -> (Cout, 8, 8). w: (Cout, 9*Cin) with k = dy*3+dx major."""
        pad[:, 1:9, 1:9] = x
        cols = np.stack(
            [
                pad[:, 0:8, 0:8],
                pad[:, 0:8, 1:9],
                pad[:, 0:8, 2:10],
                pad[:, 1:9, 0:8],
                pad[:, 1:9, 1:9],
                pad[:, 1:9, 2:10],
                pad[:, 2:10, 0:8],
                pad[:, 2:10, 1:9],
                pad[:, 2:10, 2:10],
            ]
        )  # (9, Cin, 8, 8)
        cols = cols.reshape(-1, 64)
        out = w @ cols
        out += b[:, None]
        return out.reshape(-1, 8, 8)

    def logits(self, planes: np.ndarray) -> np.ndarray:
        w = self.w
        x = self._conv3(planes, self._pad_in, w["stem_w"], w["stem_b"])
        np.maximum(x, 0.0, out=x)
        pad = self._pad
        for i in range(self.blocks):
            y = self._conv3(x, pad, w[f"block{i}_c1_w"], w[f"block{i}_c1_b"])
            np.maximum(y, 0.0, out=y)
            y = self._conv3(y, pad, w[f"block{i}_c2_w"], w[f"block{i}_c2_b"])
            y += x
            np.maximum(y, 0.0, out=y)
            x = y
        p = self._conv3(x, pad, w["pol_conv_w"], w["pol_conv_b"])
        np.maximum(p, 0.0, out=p)
        p = w["pol_out_w"] @ p.reshape(self.channels, 64) + w["pol_out_b"][:, None]  # (73, 64)
        return p.T.reshape(-1)  # index = square * 73 + plane

    def value(self, planes: np.ndarray) -> float:
        """Auxiliary value head; not used by the engine, kept for analysis."""
        w = self.w
        x = self._conv3(planes, self._pad_in, w["stem_w"], w["stem_b"])
        np.maximum(x, 0.0, out=x)
        pad = self._pad
        for i in range(self.blocks):
            y = self._conv3(x, pad, w[f"block{i}_c1_w"], w[f"block{i}_c1_b"])
            np.maximum(y, 0.0, out=y)
            y = self._conv3(y, pad, w[f"block{i}_c2_w"], w[f"block{i}_c2_b"])
            y += x
            np.maximum(y, 0.0, out=y)
            x = y
        v = w["val_conv_w"] @ x.reshape(self.channels, 64) + w["val_conv_b"][:, None]
        v = np.maximum(v, 0.0).reshape(-1)
        h = np.maximum(w["val_fc1_w"] @ v + w["val_fc1_b"], 0.0)
        return float(np.tanh(w["val_fc2_w"] @ h + w["val_fc2_b"])[0])

    def prior(self, board: chess.Board) -> dict[chess.Move, float]:
        return _prior_from_logits(board, self.logits(encode_board(board)))


class OnnxPolicy:
    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.num_params = -1

    def logits(self, planes: np.ndarray) -> np.ndarray:
        out = self.session.run(["logits"], {"planes": planes[None]})[0]
        return out[0]

    def prior(self, board: chess.Board) -> dict[chess.Move, float]:
        return _prior_from_logits(board, self.logits(encode_board(board)))


def _prior_from_logits(board: chess.Board, logits: np.ndarray) -> dict[chess.Move, float]:
    flip = not board.turn
    moves = list(board.legal_moves)
    if not moves:
        return {}
    idx = [move_to_index(m, flip) for m in moves]
    vals = logits[idx]
    m = float(vals.max())
    e = np.exp(vals - m)
    e /= e.sum()
    return dict(zip(moves, e.tolist(), strict=True))


def benchmark(policy: object, board: chess.Board, n: int = 200) -> float:
    """Median seconds per prior() call."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        policy.prior(board)  # type: ignore[attr-defined]
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def load_policy(npz_path: str, prefer: str | None = None):
    """Return a policy object or None when no weights exist."""
    prefer = prefer or os.environ.get("PN_BACKEND", "auto")
    onnx_path = os.path.splitext(npz_path)[0] + ".onnx"
    numpy_policy = NumpyPolicy(npz_path) if os.path.exists(npz_path) else None
    onnx_policy = None
    if prefer in ("auto", "onnx") and os.path.exists(onnx_path):
        try:
            onnx_policy = OnnxPolicy(onnx_path)
        except Exception as error:  # onnxruntime missing or incompatible
            print(f"onnx backend unavailable: {error!r}")
            onnx_policy = None
    if prefer == "numpy":
        return numpy_policy
    if prefer == "onnx":
        return onnx_policy or numpy_policy
    if onnx_policy is not None and numpy_policy is not None:
        board = chess.Board()
        t_np = benchmark(numpy_policy, board, 30)
        t_ox = benchmark(onnx_policy, board, 30)
        chosen = onnx_policy if t_ox < t_np else numpy_policy
        print(
            f"policy backend: numpy {t_np * 1000:.2f} ms, onnx {t_ox * 1000:.2f} ms "
            f"-> {type(chosen).__name__}"
        )
        return chosen
    return onnx_policy or numpy_policy


def logit_entropy(p: dict[chess.Move, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


__all__ = ["NUM_MOVE_PLANES", "NumpyPolicy", "OnnxPolicy", "benchmark", "load_policy"]
