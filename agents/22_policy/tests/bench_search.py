"""Search benchmark: fixed depth per position, node counts and CPU-time node rate.

CPU time (process_time) is used because the development machine is shared with many other
processes; wall time there says more about the load than about the engine.

    python tests/bench_search.py [depth] [policy:0/1]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402
from pn_search import Searcher  # noqa: E402

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "2r3k1/pp3ppp/4p3/3nP3/3P4/1Q3N2/P4PPP/1q3RK1 w - - 0 1",
    "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 1",
]


def main() -> None:
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    use_policy = len(sys.argv) > 2 and sys.argv[2] == "1"
    min_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    prior = None
    if use_policy:
        from pn_policy import load_policy

        net = load_policy(
            os.environ.get(
                "PN_MODEL_PATH", os.path.join(os.path.dirname(HERE), "models", "policy.npz")
            )
        )
        prior = net.prior if net is not None else None
        print("policy loaded:", net is not None)

    s = Searcher(prior=prior, policy_min_depth=min_depth)
    total_nodes = 0
    total_cpu = 0.0
    for fen in POSITIONS:
        board = chess.Board(fen)
        s.new_game()
        c0 = time.process_time()
        w0 = time.perf_counter()
        r = s.search(board, depth, 600.0)
        cpu = time.process_time() - c0
        wall = time.perf_counter() - w0
        st = r.stats
        total_nodes += st.nodes
        total_cpu += cpu
        print(
            f"depth {r.depth:2d} sel {st.seldepth:2d} score {r.score:6d} nodes {st.nodes:7d} "
            f"q {st.qnodes:7d} ({100 * st.qnodes / max(1, st.nodes):.0f}%) "
            f"tt {st.tt_hits:6d} pol {st.policy_calls:5d} "
            f"cpu-nps {int(st.nodes / max(cpu, 1e-6)):6d} cpu {cpu:.2f}s wall {wall:.2f}s "
            f"pv {' '.join(m.uci() for m in r.pv[:5])}"
        )
    print(
        f"TOTAL nodes {total_nodes} cpu {total_cpu:.2f}s "
        f"cpu-nps {int(total_nodes / max(total_cpu, 1e-6))}"
    )


if __name__ == "__main__":
    main()
