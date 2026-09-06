"""Where is the evaluation weak? Error against the teacher, bucketed by phase and by position type.

The network is trained on one pooled objective, so a good average hides which kind of position
it gets wrong. This splits the held-out error by game stage, by material on the board, by how
lopsided the position is and by whether material is imbalanced, which is what tells you where
more data or a different loss would pay.

    .venv-cuda/Scripts/python.exe training/analyse.py --model models/deepchess_v2.pt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from train import DeepChessNet, encode_features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# piece codes: 1..6 white PNBRQK, 7..12 black PNBRQK
PHASE_WEIGHT = {2: 1, 3: 1, 4: 2, 5: 4, 8: 1, 9: 1, 10: 2, 11: 4}
VALUE = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 7: -100, 8: -320, 9: -330, 10: -500, 11: -900}


def board_stats(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(phase 0-24, material balance in cp from White) for each position."""
    phase = np.zeros(boards.shape[0], dtype=np.int32)
    material = np.zeros(boards.shape[0], dtype=np.int32)
    for code, weight in PHASE_WEIGHT.items():
        phase += weight * (boards == code).sum(axis=1).astype(np.int32)
    for code, value in VALUE.items():
        material += value * (boards == code).sum(axis=1).astype(np.int32)
    return np.minimum(phase, 24), material


def report(name: str, mask: np.ndarray, err: np.ndarray, acc: np.ndarray) -> None:
    n = int(mask.sum())
    if n < 200:
        print(f"  {name:34} n={n:7d}  (too few)")
        return
    print(f"  {name:34} n={n:7d}  MAE {err[mask].mean():6.1f} cp   "
          f"sign agreement {acc[mask].mean():5.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "deepchess_v2.pt"))
    parser.add_argument("--data", default=str(ROOT / "data" / "*.npz"))
    parser.add_argument("--limit", type=int, default=400_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = sorted(glob.glob(args.data))[::3]  # a spread across the runs, not one tail
    boards, metas, scores, plies = [], [], [], []
    total = 0
    for path in files:
        d = np.load(path)
        boards.append(d["boards"])
        metas.append(d["meta"])
        scores.append(d["scores"])
        plies.append(d["plies"])
        total += d["boards"].shape[0]
        if total >= args.limit:
            break
    boards_a = np.concatenate(boards)[: args.limit]
    metas_a = np.concatenate(metas)[: args.limit]
    scores_a = np.concatenate(scores)[: args.limit].astype(np.float32)
    plies_a = np.concatenate(plies)[: args.limit].astype(np.int32)
    print(f"{boards_a.shape[0]} positions from {len(files)} files")

    idx = encode_features(boards_a, metas_a)
    model = DeepChessNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval().to(args.device)
    outs = []
    with torch.no_grad():
        x = torch.from_numpy(idx).to(args.device)
        for i in range(0, x.shape[0], 32768):
            outs.append(model(x[i:i + 32768].long()).cpu())
    v = torch.cat(outs).numpy()

    # calibrate V -> centipawns on this sample, exactly as training reports it
    a, b = np.polyfit(v, scores_a, 1)
    pred = a * v + b
    err = np.abs(pred - scores_a)
    acc = (np.sign(pred) == np.sign(scores_a)) | (np.abs(scores_a) < 20)
    phase, material = board_stats(boards_a)
    print(f"overall MAE {err.mean():.1f} cp, sign agreement {acc.mean():.1%}\n")

    print("by game stage (ply index within the game):")
    report("opening      ply < 20", plies_a < 20, err, acc)
    report("middlegame   20 <= ply < 60", (plies_a >= 20) & (plies_a < 60), err, acc)
    report("late middle  60 <= ply < 100", (plies_a >= 60) & (plies_a < 100), err, acc)
    report("endgame      ply >= 100", plies_a >= 100, err, acc)

    print("\nby material on the board (phase, 24 = all the pieces):")
    report("phase >= 20  (opening-like)", phase >= 20, err, acc)
    report("phase 12-19  (middlegame)", (phase >= 12) & (phase < 20), err, acc)
    report("phase 6-11   (simplified)", (phase >= 6) & (phase < 12), err, acc)
    report("phase 1-5    (late endgame)", (phase >= 1) & (phase < 6), err, acc)
    report("phase 0      (pawn endings)", phase == 0, err, acc)

    print("\nby how decided the position is (teacher score):")
    report("balanced     |cp| < 50", np.abs(scores_a) < 50, err, acc)
    report("small edge   50-150", (np.abs(scores_a) >= 50) & (np.abs(scores_a) < 150), err, acc)
    report("clear edge   150-400", (np.abs(scores_a) >= 150) & (np.abs(scores_a) < 400), err, acc)
    report("winning      400-1500", (np.abs(scores_a) >= 400) & (np.abs(scores_a) < 1500), err, acc)
    report("won          >= 1500", np.abs(scores_a) >= 1500, err, acc)

    print("\nby material imbalance (is the edge material or position?):")
    balanced_material = np.abs(material) < 100
    report("material level", balanced_material, err, acc)
    report("material imbalanced", ~balanced_material, err, acc)
    report("level material, big score", balanced_material & (np.abs(scores_a) >= 300), err, acc)

    print("\nduplicate positions in the sample:")
    packed = np.ascontiguousarray(boards_a).view(np.dtype((np.void, boards_a.shape[1])))
    unique = np.unique(packed).shape[0]
    print(f"  {boards_a.shape[0] - unique} of {boards_a.shape[0]} are repeats "
          f"({100 * (1 - unique / boards_a.shape[0]):.1f}%)")


if __name__ == "__main__":
    main()
