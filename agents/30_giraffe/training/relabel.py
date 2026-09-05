"""Relabel generated positions with a fixed-depth handcrafted search (search distillation).

The supervised stage first taught the network the quiescence-resolved static score, which
made it a noisy tactics predictor that the search then trusted at stand-pat; in the
controlled arena that net lost 4-20 to the static score alone. This script builds the
dataset the residual network is trained on instead:

* only quiet positions are kept (not in check, quiescence value equal to the static
  value), because those are the positions the search hands to the evaluator;
* the label is the score of a ``--depth`` ply alpha-beta search with the handcrafted
  evaluator, so ``label - static`` is what two more plies of search know that the static
  score does not; the network learns that as a residual and the static part stays exact.

    python training/relabel.py --data training/data/bootstrap.npz --depth 2 --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import chess
import numpy as np

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import giraffe_eval as ge  # noqa: E402
from giraffe_search import INF, MATE_BOUND, Searcher  # noqa: E402

LABEL_CLAMP = 1500


def worker(args: tuple[list[str], int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    fens, depth = args
    searcher = Searcher(ge.hce_eval)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    statics: list[int] = []
    kept: list[str] = []
    for fen in fens:
        board = chess.Board(fen)
        if board.is_check():
            continue
        static = ge.hce_eval(board)
        if searcher._quiescence(board, -INF, INF, 0) != static:
            continue  # not quiet: a capture changes the value
        _move, score = searcher.search(board, 1e9, depth)
        if abs(score) >= MATE_BOUND:
            continue
        feats.append(ge.board_features(board).astype(np.float16))
        labels.append(int(max(-LABEL_CLAMP, min(LABEL_CLAMP, score))))
        statics.append(static)
        kept.append(fen)
    if not feats:
        return np.zeros((0, ge.N_INPUT), np.float16), np.zeros(0, np.float32), np.zeros(0, np.float32), []
    return np.stack(feats), np.array(labels, np.float32), np.array(statics, np.float32), kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=AGENT_DIR / "training" / "data" / "bootstrap.npz")
    parser.add_argument("--out", type=Path, default=None, help="default training/data/search_d<depth>.npz")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="positions to consider (0 = all)")
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10, help="chunks between partial saves")
    arguments = parser.parse_args()
    out = arguments.out or AGENT_DIR / "training" / "data" / f"search_d{arguments.depth}.npz"

    fens = [str(f) for f in np.load(arguments.data)["fens"]]
    if arguments.limit:
        fens = fens[: arguments.limit]
    jobs = [(fens[i : i + arguments.chunk], arguments.depth) for i in range(0, len(fens), arguments.chunk)]
    started = time.time()
    parts = []
    done = 0
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> tuple[np.ndarray, np.ndarray]:
        # written every few chunks so a partial file is usable while the run continues
        feats = np.concatenate([p[0] for p in parts])
        labels = np.concatenate([p[1] for p in parts])
        statics = np.concatenate([p[2] for p in parts])
        kept_fens = [f for p in parts for f in p[3]]
        np.savez_compressed(out, features=feats, labels=labels, static=statics, fens=np.array(kept_fens))
        return labels, statics

    with mp.Pool(arguments.workers) as pool:
        for part in pool.imap_unordered(worker, jobs):
            parts.append(part)
            done += 1
            if done % arguments.save_every == 0 or done == len(jobs):
                labels, statics = save()
                print(f"{done}/{len(jobs)} chunks, {len(labels)} quiet positions saved, {time.time() - started:.0f}s", flush=True)
    labels, statics = save()
    residual = labels - statics
    print(
        f"wrote {len(labels)} of {len(fens)} positions to {out} in {time.time() - started:.0f}s; "
        f"residual mean {residual.mean():.1f} std {residual.std():.1f} |residual|>200: {(np.abs(residual) > 200).mean():.1%}"
    )


if __name__ == "__main__":
    main()
