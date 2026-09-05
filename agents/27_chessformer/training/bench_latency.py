"""Measure parameter count, serialized size and batch-1 CPU latency (torch vs numpy forward).

Run before committing to a configuration:
    python training/bench_latency.py
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import chess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cf_encode import encode  # noqa: E402
from cf_model import Chessformer, Config, count_params  # noqa: E402

torch.set_num_threads(1)

CONFIGS = {
    "tiny": Config(dim=64, layers=2, heads=4, mlp_ratio=2, smol_hidden=64),
    "small": Config(dim=96, layers=4, heads=4, mlp_ratio=2, smol_hidden=128),
    "medium": Config(dim=128, layers=4, heads=8, mlp_ratio=2, smol_hidden=128),
    "large": Config(dim=128, layers=6, heads=8, mlp_ratio=4, smol_hidden=256),
}


def bench(name: str, cfg: Config, n: int = 200) -> None:
    from cf_infer import NumpyChessformer

    model = Chessformer(cfg).eval()
    params = count_params(model)
    feats = torch.from_numpy(encode(chess.Board())).unsqueeze(0)
    with torch.no_grad():
        for _ in range(10):
            model(feats)
        t0 = time.perf_counter()
        for _ in range(n):
            model(feats)
        torch_ms = (time.perf_counter() - t0) * 1000 / n
    path = os.path.join(HERE, f"_bench_{name}.pt")
    torch.save({"config": cfg.as_dict(), "state_dict": model.state_dict()}, path)
    size_mb = os.path.getsize(path) / 1e6
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    net = NumpyChessformer(cfg.as_dict(), state)
    x = feats[0].numpy().astype(np.float32)
    for _ in range(10):
        net.forward(x)
    t0 = time.perf_counter()
    for _ in range(n):
        net.forward(x)
    np_ms = (time.perf_counter() - t0) * 1000 / n
    os.remove(path)
    print(
        f"{name:7s} params {params / 1e6:6.2f}M file {size_mb:5.1f}MB torch {torch_ms:6.2f}ms "
        f"numpy {np_ms:6.2f}ms"
    )


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    for name in names:
        bench(name, CONFIGS[name])
