"""Print summary statistics of generated shards."""

import glob
import os
import sys

import numpy as np


def _hist(a: np.ndarray) -> dict[int, int]:
    keys, counts = np.unique(a, return_counts=True)
    return dict(zip(keys.tolist(), counts.tolist(), strict=True))


folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
files = sorted(glob.glob(os.path.join(folder, "shard_*.npz")))
n = 0
depths = []
nm = []
values = []
results = []
plies = []
for f in files:
    d = np.load(f)
    n += len(d["meta"])
    depths.append(d["depth"])
    nm.append(d["n_moves"])
    values.append(d["value"])
    results.append(d["result"])
    plies.append(d["ply"])
if not files:
    print("no shards")
    sys.exit(0)
depths = np.concatenate(depths)
nm = np.concatenate(nm)
values = np.concatenate(values)
results = np.concatenate(results)
plies = np.concatenate(plies)
print(f"shards {len(files)} positions {n}")
print("depth histogram:", _hist(depths))
print(f"moves/position mean {nm.mean():.1f}  labels total {nm.sum()}")
print(
    f"value mean {values.mean():.0f} std {values.std():.0f}  "
    f"|v|>500: {(np.abs(values) > 500).mean():.2%}"
)
print("result histogram:", _hist(results))
print(f"ply mean {plies.mean():.1f} max {plies.max()}")
