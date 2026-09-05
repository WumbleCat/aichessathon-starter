"""Torch twin of the numba network in ``giraffe_eval`` plus flat-weight import/export.

The numba forward pass is the one that plays; this module exists to train it. The two
must agree bit-for-bit in layout, which ``tests/test_training.py`` checks by comparing
outputs on random inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import giraffe_eval as ge  # noqa: E402


class GiraffeNet(nn.Module):
    """Three input groups -> merged hidden layers -> scaled tanh output (centipawns)."""

    def __init__(self) -> None:
        super().__init__()
        self.g = nn.Linear(ge.N_GLOBAL, ge.H_G)
        self.p = nn.Linear(ge.N_PIECE, ge.H_P)
        self.s = nn.Linear(ge.N_SQUARE, ge.H_S)
        self.h2 = nn.Linear(ge.H_MERGED, ge.H_2)
        self.h3 = nn.Linear(ge.H_2, ge.H_3)
        self.out = nn.Linear(ge.H_3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns tanh(z) in [-1, 1]; multiply by OUT_SCALE for centipawns."""
        a = ge.N_GLOBAL
        b = a + ge.N_PIECE
        h = torch.cat(
            [
                torch.relu(self.g(x[:, :a])),
                torch.relu(self.p(x[:, a:b])),
                torch.relu(self.s(x[:, b:])),
            ],
            dim=1,
        )
        h = torch.relu(self.h2(h))
        h = torch.relu(self.h3(h))
        return torch.tanh(self.out(h)).squeeze(1)

    # ---------------------------------------------------------------- flat weights

    def _layers(self) -> dict[str, nn.Linear]:
        return {"g": self.g, "p": self.p, "s": self.s, "h2": self.h2, "h3": self.h3, "out": self.out}

    def to_flat(self) -> np.ndarray:
        flat = np.zeros(ge.N_WEIGHTS, dtype=np.float32)
        for name, layer in self._layers().items():
            off_w, off_b, n_out, n_in = ge.LAYOUT[name]
            flat[off_w : off_w + n_out * n_in] = layer.weight.detach().cpu().numpy().reshape(-1)
            flat[off_b : off_b + n_out] = layer.bias.detach().cpu().numpy()
        return flat

    def load_flat(self, flat: np.ndarray) -> None:
        with torch.no_grad():
            for name, layer in self._layers().items():
                off_w, off_b, n_out, n_in = ge.LAYOUT[name]
                layer.weight.copy_(torch.from_numpy(flat[off_w : off_w + n_out * n_in].reshape(n_out, n_in)))
                layer.bias.copy_(torch.from_numpy(flat[off_b : off_b + n_out]))


def save_weights(model: GiraffeNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=model.to_flat())


def load_weights(path: Path) -> np.ndarray:
    return np.load(path)["weights"].astype(np.float32)


def cp_to_target(cp: np.ndarray) -> np.ndarray:
    """Centipawns -> network output space."""
    return np.tanh(cp / ge.OUT_SCALE).astype(np.float32)
