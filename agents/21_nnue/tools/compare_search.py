"""Fixed-depth search fingerprint, to check that a refactor changes no search decision.

Runs the engine found in ``root`` (default: this agent) over a fixed set of positions with a
cleared TT and prints one line per position: fen | best move | score | depth | nodes.  Run it
once per engine version, in separate processes (numba cannot link two modules that share a
name), and diff the outputs; a pure compile-time refactor must give identical lines.

    python tools/compare_search.py --depth 6 > new.txt
    python tools/compare_search.py --root <reference agent dir> --depth 6 > old.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/P4k2/8/8/8/8/1p3K2/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "2r3k1/pp3ppp/8/3p4/3P4/8/PP3PPP/2R3K1 b - - 0 1",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT, help="agent directory holding csearch.py")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "nnue.safetensors"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    import csearch
    import nnue

    engine = csearch.Searcher(nnue.load_net(args.weights), use_nnue=True)
    total_cpu = 0.0
    for fen in FENS:
        engine.clear()
        engine.set_position(chess.Board(fen), [])
        t = time.process_time()
        move, score, depth, _pv, stats = engine.search(max_depth=args.depth)
        dt = time.process_time() - t
        total_cpu += dt
        print(f"{fen} | {move} | {score} | {depth} | {stats['nodes']}")
        print(f"# {dt:.2f} s cpu", file=sys.stderr)
    print(f"# total {total_cpu:.2f} s cpu for {len(FENS)} searches", file=sys.stderr)


if __name__ == "__main__":
    main()
