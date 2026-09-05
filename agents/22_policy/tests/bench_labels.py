"""Cost of teacher labelling (root scores for every legal move) at fixed depths."""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402
from bench_search import POSITIONS  # noqa: E402
from pn_search import Searcher  # noqa: E402

s = Searcher()
for depth in (2, 3):
    total = 0
    cpu = 0.0
    for fen in POSITIONS:
        b = chess.Board(fen)
        s.new_game()
        c0 = time.process_time()
        r = s.search(b, depth, 600.0, want_root_scores=True)
        cpu += time.process_time() - c0
        total += r.stats.nodes
        best = max(r.root_scores.values())
        exact = sum(1 for v in r.root_scores.values() if v >= best - 300)
        print(
            f"  depth {depth} nodes {r.stats.nodes:6d} "
            f"moves {len(r.root_scores):2d} exact {exact:2d}"
        )
    print(
        f"depth {depth}: total nodes {total} cpu {cpu:.2f}s -> {cpu / len(POSITIONS):.2f}s/position"
    )
