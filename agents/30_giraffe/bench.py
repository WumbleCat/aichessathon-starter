"""Benchmarks for RESULTS.md: evaluator latency, search speed and depth, model size.

    python bench.py [--model models/giraffe.npz] [--budget 2.0]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import giraffe_eval as ge  # noqa: E402
from giraffe_search import Searcher  # noqa: E402

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
]


def eval_latency(evaluate: ge.Evaluator, n: int = 20000) -> float:
    board = chess.Board(POSITIONS[2])
    evaluate(board)
    started = time.perf_counter()
    for _ in range(n):
        evaluate(board)
    return (time.perf_counter() - started) / n * 1e6


def search_bench(evaluate: ge.Evaluator, budget: float) -> dict[str, float]:
    depths: list[int] = []
    seldepths: list[int] = []
    nodes = qnodes = tt_hits = 0
    elapsed = 0.0
    for fen in POSITIONS:
        searcher = Searcher(evaluate)
        searcher.search(chess.Board(fen), budget, 64)
        s = searcher.stats
        depths.append(s.depth)
        seldepths.append(s.seldepth)
        nodes += s.nodes
        qnodes += s.qnodes
        tt_hits += s.tt_hits
        elapsed += s.elapsed
    return {
        "depth_mean": statistics.mean(depths),
        "depth_min": min(depths),
        "depth_max": max(depths),
        "seldepth_mean": statistics.mean(seldepths),
        "nodes": nodes,
        "qnodes": qnodes,
        "qnode_share": qnodes / max(1, nodes),
        "tt_hits": tt_hits,
        "nps": nodes / max(elapsed, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=HERE / "models" / "giraffe.npz")
    parser.add_argument("--budget", type=float, default=2.0)
    arguments = parser.parse_args()
    ge.warm_up()

    print(f"network: {ge.N_WEIGHTS} parameters, input {ge.N_INPUT} features")
    if arguments.model.exists():
        weights = np.load(arguments.model)["weights"].astype(np.float32)
        print(f"model file: {arguments.model.name} {arguments.model.stat().st_size / 1024:.0f} KB")
        evaluators = [("net", ge.NetEvaluator(weights)), ("hce", ge.hce_eval)]
    else:
        print("no model file; benchmarking the handcrafted evaluator only")
        evaluators = [("hce", ge.hce_eval)]

    for name, evaluate in evaluators:
        print(f"\n[{name}] batch-1 latency {eval_latency(evaluate):.1f} us per position (board -> centipawns)")
        stats = search_bench(evaluate, arguments.budget)
        print(
            f"[{name}] {arguments.budget:.1f}s/move over {len(POSITIONS)} positions: depth {stats['depth_mean']:.1f} "
            f"(min {stats['depth_min']}, max {stats['depth_max']}), seldepth {stats['seldepth_mean']:.1f}, "
            f"nodes {stats['nodes']}, qnodes {stats['qnodes']} ({stats['qnode_share']:.0%}), "
            f"tt_hits {stats['tt_hits']}, {stats['nps']:.0f} nodes/s"
        )


if __name__ == "__main__":
    main()
