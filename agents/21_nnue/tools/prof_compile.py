"""Where does the numba compile time go?

Runs the same warm-up as ``agent._build_engine`` in this process, then lists every jitted
function of cboard/csearch/nnue with its number of compiled overloads and the compile pipeline
time numba recorded for each.  CPU seconds are the meaningful figure on a loaded machine.

    python tools/prof_compile.py            # from agents/21_nnue
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main() -> None:
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    import chess

    import cboard
    import csearch
    import nnue

    net_path = nnue.default_weights_path()
    net = nnue.load_net(net_path) if os.path.exists(net_path) else None
    searcher = csearch.Searcher(net, use_nnue=net is not None)
    fens = [
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P4k2/8/8/8/8/1p3K2/8 w - - 0 1",
    ]
    for fen in fens:
        searcher.set_position(chess.Board(fen), [])
        searcher.search(max_depth=3, node_limit=3000)
    wall = time.perf_counter() - t_wall
    cpu = time.process_time() - t_cpu
    print(f"build: {wall:.1f} s wall, {cpu:.1f} s cpu")

    rows: list[tuple[float, str, int, str]] = []
    for mod in (cboard, csearch, nnue):
        for name in dir(mod):
            fn = getattr(mod, name)
            overloads = getattr(fn, "overloads", None)
            if not isinstance(overloads, dict) or not overloads:
                continue
            total = 0.0
            detail = []
            for cres in overloads.values():
                times = cres.metadata.get("pipeline_times", {})
                for _pipe, passes in times.items():
                    for pass_name, pt in passes.items():
                        total += pt.run
                        detail.append((pt.run, pass_name))
            detail.sort(reverse=True)
            top = ", ".join(f"{p}={t:.1f}" for t, p in detail[:3])
            rows.append((total, f"{mod.__name__}.{name}", len(overloads), top))
    rows.sort(reverse=True)
    print(f"{'seconds':>8}  {'ovl':>3}  function  (top passes)")
    for total, name, n, top in rows:
        print(f"{total:8.2f}  {n:3d}  {name}  ({top})")
    print(f"sum of per-function pipeline times: {sum(r[0] for r in rows):.1f} s")


if __name__ == "__main__":
    main()
