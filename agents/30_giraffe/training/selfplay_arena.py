"""In-process arena between two evaluators driving the identical search.

This is the controlled experiment the architecture README asks for: the search is held
fixed and only the leaf evaluator changes. Games are played in pairs from the same random
opening with colours swapped, at a fixed per-move time budget or fixed depth, across a
process pool. It is much cheaper than the harness (no per-game interpreter start) and is
also used to gate TD-Leaf checkpoints.

    python training/selfplay_arena.py --a models/giraffe.npz --b hce --pairs 20 --budget 0.2
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import giraffe_eval as ge  # noqa: E402
from giraffe_search import Searcher  # noqa: E402

PLY_CAP = 240
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def make_evaluator(spec: str | np.ndarray) -> ge.Evaluator:
    if isinstance(spec, np.ndarray):
        return ge.NetEvaluator(spec)
    if spec == "hce":
        return ge.hce_eval
    return ge.NetEvaluator(np.load(spec)["weights"].astype(np.float32))


def random_opening(rng: random.Random, plies: int) -> str:
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board.fen()


def adjudicate(board: chess.Board) -> str:
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    return "1-0" if balance > 0 else "0-1" if balance < 0 else "1/2-1/2"


def play(white: ge.Evaluator, black: ge.Evaluator, fen: str, budget: float, depth: int) -> str:
    board = chess.Board(fen)
    searchers = {chess.WHITE: Searcher(white), chess.BLACK: Searcher(black)}
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return outcome.result()
        if len(board.move_stack) >= PLY_CAP:
            return adjudicate(board)
        searcher = searchers[board.turn]
        searcher.remember(board)
        move, _ = searcher.search(board, budget, depth)
        board.push(move)
        searcher.remember(board)


def pair(args: tuple[object, object, str, float, int]) -> tuple[float, float]:
    """Plays one opening with both colours; returns A's score in each game."""
    spec_a, spec_b, fen, budget, depth = args
    a = make_evaluator(spec_a)  # type: ignore[arg-type]
    b = make_evaluator(spec_b)  # type: ignore[arg-type]
    first = play(a, b, fen, budget, depth)
    second = play(b, a, fen, budget, depth)
    score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    return score[first], 1.0 - score[second]


def run(
    spec_a: object,
    spec_b: object,
    pairs: int,
    budget: float,
    depth: int,
    workers: int,
    seed: int,
    opening_plies: int = 6,
    pool: mp.pool.Pool | None = None,
) -> tuple[int, int, int]:
    rng = random.Random(seed)
    jobs = [(spec_a, spec_b, random_opening(rng, opening_plies), budget, depth) for _ in range(pairs)]
    if pool is None:
        with mp.Pool(workers) as own_pool:
            results = own_pool.map(pair, jobs)
    else:
        results = pool.map(pair, jobs)
    wins = draws = losses = 0
    for first, second in results:
        for score in (first, second):
            if score == 1.0:
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1
    return wins, draws, losses


def elo(wins: int, draws: int, losses: int) -> str:
    games = wins + draws + losses
    if games == 0:
        return "n/a"
    p = (wins + draws / 2) / games
    p = min(max(p, 1e-3), 1 - 1e-3)
    diff = -400 * math.log10(1 / p - 1)
    # standard error of the score, propagated through the logistic
    se = math.sqrt(max(p * (1 - p), 1e-6) / games)
    return f"{diff:+.0f} elo (score {p:.1%} over {games} games, +/- {400 * se / (p * (1 - p) * math.log(10)):.0f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=str(AGENT_DIR / "models" / "giraffe.npz"))
    parser.add_argument("--b", default="hce")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--budget", type=float, default=0.2, help="seconds per move")
    parser.add_argument("--depth", type=int, default=64, help="max depth (use with a huge budget for fixed depth)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opening-plies", type=int, default=6)
    arguments = parser.parse_args()
    started = time.time()
    wins, draws, losses = run(
        arguments.a, arguments.b, arguments.pairs, arguments.budget, arguments.depth, arguments.workers, arguments.seed, arguments.opening_plies
    )
    print(f"A={arguments.a} vs B={arguments.b}: +{wins} ={draws} -{losses}  {elo(wins, draws, losses)}  [{time.time() - started:.0f}s]")


if __name__ == "__main__":
    main()
