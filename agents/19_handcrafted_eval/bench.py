"""Benchmarks for agent 19.

    python bench.py eval                      # evaluation micro-benchmark
    python bench.py search [--ms 1000]        # nodes/s, depth, eval calls on a few positions
    python bench.py depth [--depth 4]         # fixed-depth node counts (load independent)
    python bench.py arena --opponent baselines/greedy --games 20 [--base-ms 10000 --inc-ms 100]

`arena` plays paired games: every opening is played once with our agent as White and once as
Black, through the official harness sandbox and clock rules, and records per-move timing and
the search statistics the agent prints when HCE_INFO=1 is set.

Run from the repository root so that `harness` imports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/pp1n1ppp/2pbpq2/3p4/3P1B2/2NBPN2/PPQ2PPP/R3K2R w KQkq - 0 10",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 7",
    "2r3k1/1p3ppp/p7/8/8/1P6/P4PPP/3R2K1 w - - 0 1",
    "8/5k2/3p4/8/8/8/3K1P2/4R3 w - - 0 1",
    "r2q1rk1/1b1nbppp/pp2pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/R2Q1RK1 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
]

# neutral openings for paired games, as SAN sequences from the start position
OPENINGS = [
    "e4 e5 Nf3 Nc6",
    "d4 d5 c4 e6",
    "e4 c5 Nf3 d6",
    "d4 Nf6 c4 g6",
    "e4 e6 d4 d5",
    "c4 e5 Nc3 Nf6",
    "Nf3 d5 g3 Nf6",
    "e4 c6 d4 d5",
    "d4 d5 Nf3 Nf6 c4 c6",
    "e4 e5 Nf3 Nf6",
    "d4 Nf6 c4 e6 Nc3 Bb4",
    "e4 d6 d4 Nf6",
    "e4 e5 Bc4 Nf6",
    "d4 f5 g3 Nf6",
    "c4 c5 Nf3 Nf6",
    "e4 g6 d4 Bg7",
]


def opening_fens() -> list[str]:
    fens = []
    for line in OPENINGS:
        board = chess.Board()
        for san in line.split():
            board.push_san(san)
        fens.append(board.fen())
    return fens


def bench_eval() -> None:
    import hce_eval

    print(f"numba: {hce_eval.USING_NUMBA}")
    for fen in POSITIONS:
        board = chess.Board(fen)
        hce_eval.evaluate_stm(board)
        n = 20000
        started = time.perf_counter()
        for _ in range(n):
            hce_eval.evaluate_stm(board)
        per_call = (time.perf_counter() - started) / n * 1e6
        print(f"{per_call:6.2f} us  score {hce_eval.evaluate_board(board):6d}  {fen}")


def bench_search(ms: int) -> None:
    from hce_eval import evaluate_stm
    from hce_search import Searcher

    total_nodes = 0
    total_time = 0.0
    depths = []
    for fen in POSITIONS:
        board = chess.Board(fen)
        searcher = Searcher(evaluate_stm)
        started = time.monotonic()
        move, score = searcher.search_root(board, started + ms / 1000, started + ms / 1000)
        elapsed = time.monotonic() - started
        total_nodes += searcher.nodes
        total_time += elapsed
        depths.append(searcher.depth_reached)
        print(
            f"{move.uci():6} score {score:6d} "
            f"depth {searcher.depth_reached:2d}/{searcher.seldepth:2d} "
            f"nodes {searcher.nodes:7d} qnodes {searcher.qnodes:7d} "
            f"evals {searcher.eval_calls:7d} tt_hits {searcher.tt_hits:6d} "
            f"nps {searcher.nodes / elapsed:7.0f}  {fen}"
        )
    print(
        f"\ntotal nodes {total_nodes} in {total_time:.1f}s = {total_nodes / total_time:.0f} nps, "
        f"mean depth {statistics.mean(depths):.1f}"
    )


def bench_depth(depth: int) -> None:
    """Fixed-depth search: node counts are independent of machine load, so this is the
    yardstick for ordering/pruning changes."""
    from hce_eval import evaluate_stm
    from hce_search import Searcher

    total_nodes = 0
    total_evals = 0
    total_time = 0.0
    for fen in POSITIONS:
        board = chess.Board(fen)
        searcher = Searcher(evaluate_stm)
        started = time.perf_counter()
        move, score = searcher.search_root(board, float("inf"), float("inf"), depth)
        elapsed = time.perf_counter() - started
        total_nodes += searcher.nodes
        total_evals += searcher.eval_calls
        total_time += elapsed
        print(
            f"{move.uci():6} score {score:6d} nodes {searcher.nodes:8d} "
            f"qnodes {searcher.qnodes:8d} evals {searcher.eval_calls:8d} "
            f"tt_hits {searcher.tt_hits:6d} seldepth {searcher.seldepth:2d} "
            f"{elapsed * 1000:7.0f}ms  {fen}"
        )
    print(
        f"\ndepth {depth}: total nodes {total_nodes} evals {total_evals} "
        f"time {total_time:.1f}s nps {total_nodes / max(total_time, 1e-9):.0f}"
    )


INFO_RE = re.compile(
    r"info depth (\d+) seldepth (\d+) nodes (\d+) qnodes (\d+) evals (\d+) tt_hits (\d+) "
    r"time (\d+)ms"
)


def bench_arena(opponent: str, games: int, base_ms: int, inc_ms: int, out: str | None) -> None:
    from harness.referee import FAILED_TERMINATIONS
    from harness.rules import INIT_BUDGET_S, PLY_CAP
    from harness.sandbox import AgentFailure, local

    os.environ["HCE_INFO"] = "1"
    fens = opening_fens()
    agent_dir = HERE
    opp_dir = (ROOT / opponent).resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    move_times: list[float] = []
    depths: list[int] = []
    nodes_total = 0
    qnodes_total = 0
    evals_total = 0
    tt_hits_total = 0
    search_time_total = 0.0
    seldepths: list[int] = []
    records = []

    for game in range(games):
        fen = fens[(game // 2) % len(fens)]
        plays_white = game % 2 == 0
        white = local(agent_dir if plays_white else opp_dir)
        black = local(opp_dir if plays_white else agent_dir)
        agents = {chess.WHITE: white, chess.BLACK: black}
        mine = chess.WHITE if plays_white else chess.BLACK
        board = chess.Board(fen)
        clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
        result = "draw"
        termination = "unknown"
        try:
            failed = None
            for colour, proc in agents.items():
                try:
                    proc.start(INIT_BUDGET_S)
                except AgentFailure as failure:
                    failed = (colour, failure.reason)
                    break
            if failed is not None:
                colour, reason = failed
                result = "black" if colour == chess.WHITE else "white"
                termination = reason
            else:
                while True:
                    finish = board.outcome(claim_draw=True)
                    if finish is not None:
                        termination = finish.termination.name.lower()
                        result = (
                            "draw" if finish.winner is None
                            else ("white" if finish.winner == chess.WHITE else "black")
                        )
                        break
                    if len(board.move_stack) >= PLY_CAP:
                        termination = "adjudication"
                        balance = sum(
                            v * (len(board.pieces(p, True)) - len(board.pieces(p, False)))
                            for p, v in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 9))
                        )
                        result = "white" if balance > 0 else "black" if balance < 0 else "draw"
                        break
                    mover = board.turn
                    started = time.monotonic()
                    try:
                        uci = agents[mover].move(board.fen(), int(clock[mover]))
                    except AgentFailure as failure:
                        result = "black" if mover == chess.WHITE else "white"
                        termination = failure.reason
                        break
                    spent = (time.monotonic() - started) * 1000.0
                    clock[mover] -= spent
                    if mover == mine:
                        move_times.append(spent)
                    if clock[mover] < 0:
                        result = "black" if mover == chess.WHITE else "white"
                        termination = "flag"
                        break
                    try:
                        move = chess.Move.from_uci(uci)
                    except chess.InvalidMoveError:
                        move = None
                    if move is None or move not in board.legal_moves:
                        result = "black" if mover == chess.WHITE else "white"
                        termination = "illegal"
                        break
                    board.push(move)
                    clock[mover] += inc_ms
        finally:
            white.stop()
            black.stop()
        # harvest the search statistics our agent printed
        for line in agents[mine].stderr_tail.splitlines():
            m = INFO_RE.search(line)
            if m:
                d, sd, n, qn, ev, tth, ms = (int(x) for x in m.groups())
                depths.append(d)
                seldepths.append(sd)
                nodes_total += n
                qnodes_total += qn
                evals_total += ev
                tt_hits_total += tth
                search_time_total += ms / 1000.0
        terminations[termination] = terminations.get(termination, 0) + 1
        if result == "draw":
            draws += 1
        elif (result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        records.append({"game": game + 1, "fen": fen, "white": plays_white, "result": result,
                        "termination": termination, "plies": len(board.move_stack)})
        print(
            f"game {game + 1}/{games} ({'W' if plays_white else 'B'}): {result} by {termination} "
            f"in {len(board.move_stack)} plies   running +{wins} ={draws} -{losses}", flush=True
        )

    score = (wins + draws / 2) / games if games else 0
    print(f"\n{agent_dir.name} vs {opponent} over {games} games at {base_ms}+{inc_ms} ms")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in terminations.items()))
    failures = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    print(f"failures (illegal/crash/flag/init): {failures or 'none'}")
    if move_times:
        move_times.sort()
        print(
            f"move time ms: mean {statistics.mean(move_times):.0f} median "
            f"{move_times[len(move_times) // 2]:.0f} "
            f"p99 {move_times[int(len(move_times) * 0.99)]:.0f} "
            f"max {move_times[-1]:.0f}"
        )
    if depths:
        print(
            f"search: mean depth {statistics.mean(depths):.2f} mean seldepth "
            f"{statistics.mean(seldepths):.1f} nodes {nodes_total} qnodes {qnodes_total} "
            f"({qnodes_total / max(1, nodes_total):.0%}) evals {evals_total} "
            f"tt_hits {tt_hits_total} "
            f"nps {nodes_total / max(1e-9, search_time_total):.0f}"
        )
    if score and games:
        import math

        p = min(max(score, 0.001), 0.999)
        print(f"elo diff estimate: {-400 * math.log10(1 / p - 1):+.0f}")
    if out:
        Path(out).write_text(json.dumps({
            "opponent": opponent, "games": games, "base_ms": base_ms, "inc_ms": inc_ms,
            "wins": wins, "draws": draws, "losses": losses, "score": score,
            "terminations": terminations, "records": records,
            "move_time_ms": {"mean": statistics.mean(move_times) if move_times else None,
                             "max": max(move_times) if move_times else None},
            "mean_depth": statistics.mean(depths) if depths else None,
            "nodes": nodes_total, "qnodes": qnodes_total, "evals": evals_total,
            "nps": nodes_total / max(1e-9, search_time_total),
        }, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eval")
    s = sub.add_parser("search")
    s.add_argument("--ms", type=int, default=1000)
    d = sub.add_parser("depth")
    d.add_argument("--depth", type=int, default=4)
    a = sub.add_parser("arena")
    a.add_argument("--opponent", default="baselines/greedy")
    a.add_argument("--games", type=int, default=20)
    a.add_argument("--base-ms", type=int, default=10_000)
    a.add_argument("--inc-ms", type=int, default=100)
    a.add_argument("--out")
    args = parser.parse_args()
    if args.command == "eval":
        bench_eval()
    elif args.command == "search":
        bench_search(args.ms)
    elif args.command == "depth":
        bench_depth(args.depth)
    else:
        bench_arena(args.opponent, args.games, args.base_ms, args.inc_ms, args.out)


if __name__ == "__main__":
    main()
