import cProfile
import os
import pstats
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402
from pn_search import Searcher  # noqa: E402

fen = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
)
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
s = Searcher()
board = chess.Board(fen)
pr = cProfile.Profile()
pr.enable()
r = s.search(board, depth, 60.0)
pr.disable()
print(
    "nodes",
    r.stats.nodes,
    "q",
    r.stats.qnodes,
    "time",
    r.elapsed,
    "nps",
    int(r.stats.nodes / r.elapsed),
)
pstats.Stats(pr).sort_stats("tottime").print_stats(22)
