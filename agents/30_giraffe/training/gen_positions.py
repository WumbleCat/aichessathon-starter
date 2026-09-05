"""Generate the supervised bootstrap dataset (Giraffe stage 2).

Positions come from noisy self-play between one-ply handcrafted-evaluation players (a
random legal move with probability ``--random``, otherwise the greedy move with a little
evaluation noise). Every position is labelled with the quiescence-resolved handcrafted
evaluation from the side to move's point of view, so the network is first taught the
control evaluator's knowledge before TD-Leaf refines it.

Usage (from the agent directory, with the project interpreter):

    python training/gen_positions.py --positions 300000 --workers 6 \
        --out training/data/bootstrap.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import chess  # noqa: E402
import giraffe_eval as ge  # noqa: E402
from giraffe_search import INF, Searcher  # noqa: E402

MAX_PLIES = 180
SKIP_PLIES = 4
LABEL_CLAMP = 1500


def greedy_move(board: chess.Board, rng: random.Random, noise: int) -> chess.Move:
    best: chess.Move | None = None
    best_score = -(10**9)
    for move in board.legal_moves:
        board.push(move)
        score = -ge.hce_eval(board) + rng.randint(-noise, noise)
        if board.is_checkmate():
            score = 10**6
        board.pop()
        if score > best_score:
            best, best_score = move, score
    assert best is not None
    return best


def play_game(rng: random.Random, p_random: float, noise: int) -> list[str]:
    board = chess.Board()
    fens: list[str] = []
    # a few random opening plies for diversity
    opening = rng.randint(2, 8)
    for ply in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break
        if ply >= SKIP_PLIES:
            fens.append(board.fen())
        if ply < opening or rng.random() < p_random:
            move = rng.choice(list(board.legal_moves))
        else:
            move = greedy_move(board, rng, noise)
        board.push(move)
    return fens


def label(searcher: Searcher, board: chess.Board) -> int:
    searcher.stats.nodes = 0
    score = searcher._quiescence(board, -INF, INF, 0)
    return int(max(-LABEL_CLAMP, min(LABEL_CLAMP, score)))


def worker(args: tuple[int, int, float, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    seed, count, p_random, noise = args
    rng = random.Random(seed)
    searcher = Searcher(ge.hce_eval)
    feats = np.zeros((count, ge.N_INPUT), dtype=np.float16)
    labels = np.zeros(count, dtype=np.float32)
    fens: list[str] = []
    n = 0
    while n < count:
        for fen in play_game(rng, p_random, noise):
            if n >= count:
                break
            board = chess.Board(fen)
            feats[n] = ge.board_features(board)
            labels[n] = label(searcher, board)
            fens.append(fen)
            n += 1
    return feats, labels, fens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--random", type=float, default=0.25, help="probability of a random move")
    parser.add_argument(
        "--noise", type=int, default=40, help="evaluation noise in cp for greedy moves"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=AGENT_DIR / "training" / "data" / "bootstrap.npz"
    )
    arguments = parser.parse_args()

    per_worker = arguments.positions // arguments.workers
    jobs = [
        (arguments.seed * 1000 + i, per_worker, arguments.random, arguments.noise)
        for i in range(arguments.workers)
    ]
    started = time.time()
    with mp.Pool(arguments.workers) as pool:
        parts = pool.map(worker, jobs)
    feats = np.concatenate([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    fens = [f for p in parts for f in p[2]]
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.out, features=feats, labels=labels, fens=np.array(fens))
    print(
        f"wrote {len(labels)} positions to {arguments.out} in {time.time() - started:.0f}s; "
        f"label mean {labels.mean():.1f} std {labels.std():.1f} "
        f"|label|>800: {(np.abs(labels) > 800).mean():.1%}"
    )


if __name__ == "__main__":
    main()
