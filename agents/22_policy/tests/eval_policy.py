"""Offline policy metrics on held-out shards, using the engine's own numpy inference path.

Reports top-1 / top-3 / top-5 accuracy against the teacher's best move, the soft cross-entropy,
the teacher score lost by playing the network's first choice, all of it split by game phase,
plus the same numbers for the hand-crafted ordering the engine uses when there is no network
(MVV-LVA captures and promotions first, then the first quiet move) as a reference.

The validation shards are the ones ``training/train.py`` held out (same seed, same shuffle).

    python tests/eval_policy.py [--model models/policy.npz] [--seed 0] [--limit 20000]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one core, like the agent; many threads only lose time

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import chess  # noqa: E402
from pn_encoding import move_to_index  # noqa: E402
from pn_policy import NumpyPolicy  # noqa: E402
from pn_search import _mvv_lva  # noqa: E402

PIECE_CHAR = ".PNBRQKpnbrqk"


def board_from_codes(codes: np.ndarray, meta: int) -> chess.Board:
    board = chess.Board(None)
    for sq in range(64):
        c = int(codes[sq])
        if c:
            board.set_piece_at(sq, chess.Piece(c if c <= 6 else c - 6, c <= 6))
    board.turn = bool(meta & 1)
    rights = 0
    if (meta >> 1) & 1:
        rights |= chess.BB_H1
    if (meta >> 2) & 1:
        rights |= chess.BB_A1
    if (meta >> 3) & 1:
        rights |= chess.BB_H8
    if (meta >> 4) & 1:
        rights |= chess.BB_A8
    board.castling_rights = rights
    ep_file = (meta >> 5) & 15
    if ep_file < 8:
        board.ep_square = chess.square(ep_file, 5 if board.turn else 2)
    return board


def _hand_key(board: chess.Board, move: chess.Move) -> int:
    s = _mvv_lva(board, move) * 8
    if move.promotion == chess.QUEEN:
        s += 900
    return s


def phase_of(board: chess.Board) -> str:
    non_pawn = sum(
        len(board.pieces(p, c))
        for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for c in (True, False)
    )
    if non_pawn >= 12:
        return "opening"
    if non_pawn >= 6:
        return "middlegame"
    return "endgame"


def val_files(data_dir: str, seed: int, val_fraction: float) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_fraction))
    return files[:n_val]


class Tally:
    def __init__(self) -> None:
        self.n = 0
        self.top1 = 0
        self.top3 = 0
        self.top5 = 0
        self.ce = 0.0
        self.loss_cp = 0.0
        self.h_top1 = 0
        self.h_loss_cp = 0.0

    def row(self, name: str) -> str:
        n = max(self.n, 1)
        return (
            f"{name:11s} n {self.n:6d}  top1 {self.top1 / n:.3f}  top3 {self.top3 / n:.3f}  "
            f"top5 {self.top5 / n:.3f}  ce {self.ce / n:.3f}  lost-cp {self.loss_cp / n:6.1f}  |  "
            f"handcrafted top1 {self.h_top1 / n:.3f} lost-cp {self.h_loss_cp / n:6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "models", "policy.npz"))
    parser.add_argument("--data", default=os.path.join(ROOT, "training", "data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()

    policy = NumpyPolicy(args.model)
    print(
        f"model {args.model}: channels {policy.channels} blocks {policy.blocks} "
        f"params {policy.num_params:,}"
    )
    files = val_files(args.data, args.seed, args.val_fraction)
    print(f"validation shards: {len(files)}")

    tallies: dict[str, Tally] = {"all": Tally()}
    seen = 0
    t_infer = 0.0
    for f in files:
        d = np.load(f)
        keep = np.nonzero(d["depth"] >= args.min_depth)[0]
        offsets = np.zeros(len(d["meta"]) + 1, dtype=np.int64)
        np.cumsum(d["n_moves"], out=offsets[1:])
        for i in keep:
            if seen >= args.limit:
                break
            board = board_from_codes(d["codes"][i], int(d["meta"][i]))
            flip = not board.turn
            legal = list(board.legal_moves)
            if not legal:
                continue
            acts = d["label_action"][offsets[i] : offsets[i + 1]].astype(np.int64)
            scores = d["label_score"][offsets[i] : offsets[i + 1]].astype(np.float64)
            best_action = int(d["best"][i])
            best_score = float(scores.max())
            score_of = dict(zip(acts.tolist(), scores.tolist(), strict=True))

            t0 = time.perf_counter()
            prior = policy.prior(board)
            t_infer += time.perf_counter() - t0
            ranked = sorted(prior.items(), key=lambda kv: kv[1], reverse=True)
            ranked_actions = [move_to_index(m, flip) for m, _ in ranked]

            # soft cross-entropy of the network distribution against softmax(score / T)
            w = np.exp((scores - best_score) / args.temperature)
            w /= w.sum()
            prior_by_action = {move_to_index(m, flip): p for m, p in prior.items()}
            ce = -sum(
                float(wi) * math.log(max(prior_by_action.get(int(a), 0.0), 1e-12))
                for a, wi in zip(acts, w, strict=True)
            )

            # hand-crafted ordering: captures and promotions by MVV-LVA, then generation order
            hand_first = max(legal, key=lambda m, b=board: _hand_key(b, m))
            hand_action = move_to_index(hand_first, flip)

            phase = phase_of(board)
            tallies.setdefault(phase, Tally())
            for t in (tallies["all"], tallies[phase]):
                t.n += 1
                t.top1 += ranked_actions[0] == best_action
                t.top3 += best_action in ranked_actions[:3]
                t.top5 += best_action in ranked_actions[:5]
                t.ce += ce
                t.loss_cp += best_score - score_of.get(ranked_actions[0], best_score - 300)
                t.h_top1 += hand_action == best_action
                t.h_loss_cp += best_score - score_of.get(hand_action, best_score - 300)
            seen += 1
        if seen >= args.limit:
            break

    for name in ("all", "opening", "middlegame", "endgame"):
        if name in tallies:
            print(tallies[name].row(name))
    print(f"inference: {1000 * t_infer / max(seen, 1):.2f} ms per position (wall, loaded machine)")


if __name__ == "__main__":
    main()
