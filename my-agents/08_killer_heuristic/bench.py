"""Node-count benchmark: the same fixed-depth search with killers off and on.

Run from the repo root:

    uv run python my-agents/08_killer_heuristic/bench.py [depth] [--no-hash]

--no-hash switches the hash-move ordering off in both runs, which isolates what the
killers contribute on top of MVV-LVA alone. Reports nodes, cutoffs, first-move cutoff
rate, elapsed time and the chosen move for each configuration. The scores must match;
the node counts should drop with killers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess

import agent

POSITIONS = {
    "start": chess.STARTING_FEN,
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "midgame": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "queens": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
}


def run(depth: int, use_hash_move: bool) -> None:
    total = {False: 0, True: 0}
    print(f"depth {depth}, hash move {'on' if use_hash_move else 'off'}")
    print(f"{'position':10s} {'killers':8s} {'nodes':>9s} {'cutoffs':>8s} {'1st%':>6s} "
          f"{'killer%':>8s} {'time':>7s}  move    score")
    for name, fen in POSITIONS.items():
        scores: list[float] = []
        for use_killers in (False, True):
            searcher = agent.Searcher(use_killers=use_killers, use_hash_move=use_hash_move)
            _move, score = searcher.search_fixed_depth(chess.Board(fen), depth)
            s = searcher.stats
            scores.append(score)
            total[use_killers] += s.nodes
            first = 100.0 * s.first_move_cutoffs / max(1, s.cutoffs)
            killer = 100.0 * s.killer_cutoffs / max(1, s.cutoffs)
            print(
                f"{name:10s} {'on' if use_killers else 'off':8s} {s.nodes:9d} {s.cutoffs:8d} "
                f"{first:6.1f} {killer:8.1f} {s.elapsed_s:6.2f}s  {s.best_move:6s} {score:6.0f}"
            )
        assert scores[0] == scores[1], f"{name}: killers changed the score {scores}"
    saving = 100.0 * (1 - total[True] / max(1, total[False]))
    print(f"\ntotal nodes: off {total[False]}  on {total[True]}  saving {saving:.1f}%")


if __name__ == "__main__":
    depths = [int(a) for a in sys.argv[1:] if a.isdigit()]
    run(depths[0] if depths else 4, use_hash_move="--no-hash" not in sys.argv)
