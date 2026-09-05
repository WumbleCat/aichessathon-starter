"""Paired arena: every opening is played twice with colours swapped.

Uses the project's harness (same protocol, clock and referee as the platform). Openings are
random short lines from the start position generated with a fixed seed so that runs are
comparable. Engine stderr lines of the form ``depth D ... nodes N ... t Ts`` are parsed to
report average depth, nodes and move time for the agent under test.

    python tests/paired_arena.py --agent agents/22_policy \
        --opponent agents/22_policy/variants/nopolicy \
        --pairs 20 --base-ms 10000 --increment-ms 100 --jobs 4 --out results/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, PROJECT)

import chess  # noqa: E402

from harness.referee import play_match  # noqa: E402
from harness.sandbox import local  # noqa: E402

LINE = re.compile(
    r"depth (\d+) sel (\d+) score (-?\d+) nodes (\d+) q (\d+) tt (\d+) pol (\d+) "
    r"nps (\d+) t ([\d.]+)s"
)


def openings(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.choice([4, 5, 6, 7, 8])):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.is_game_over() or b.is_check():
            continue
        out.append(b.fen())
    return out


def parse_stats(text: str) -> dict:
    depths, nodes, times, pols = [], [], [], []
    for m in LINE.finditer(text):
        depths.append(int(m.group(1)))
        nodes.append(int(m.group(4)))
        pols.append(int(m.group(7)))
        times.append(float(m.group(9)))
    return {"moves": len(depths), "depth": depths, "nodes": nodes, "time": times, "pol": pols}


def _line(i: int, n: int, r: dict) -> str:
    colour = "white" if r["agent_white"] else "black"
    return f"game {i}/{n}: {r['result']} by {r['termination']} (agent {colour})"


def play_one(job: tuple) -> dict:
    agent_dir, opp_dir, fen, agent_white, base_ms, inc_ms = job
    white = local(Path(agent_dir if agent_white else opp_dir))
    black = local(Path(opp_dir if agent_white else agent_dir))
    outcome = play_match(white, black, base_ms, inc_ms, start_fen=fen)
    mine = white if agent_white else black
    stats = parse_stats(mine.stderr_tail)
    if outcome.termination in ("crash", "illegal", "init", "both_failed"):
        print("--- white stderr tail:\n" + white.stderr_tail[-1500:], file=sys.stderr)
        print("--- black stderr tail:\n" + black.stderr_tail[-1500:], file=sys.stderr)
    if outcome.result == "draw" or outcome.result == "void":
        score = 0.5
    elif (outcome.result == "white") == agent_white:
        score = 1.0
    else:
        score = 0.0
    loser_failed = outcome.termination in ("crash", "illegal", "flag", "init")
    my_failure = loser_failed and score == 0.0
    return {
        "fen": fen,
        "agent_white": agent_white,
        "result": outcome.result,
        "termination": outcome.termination,
        "score": score,
        "my_failure": my_failure,
        "plies": outcome.pgn.count(".") if outcome.pgn else 0,
        "stats": stats,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--base-ms", type=int, default=10_000)
    p.add_argument("--increment-ms", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", default="")
    args = p.parse_args()

    fens = openings(args.pairs, args.seed)
    agent_dir = os.path.abspath(os.path.join(PROJECT, args.agent))
    opp_dir = os.path.abspath(os.path.join(PROJECT, args.opponent))
    for d in (agent_dir, opp_dir):
        if not os.path.exists(os.path.join(d, "agent.py")):
            raise SystemExit(f"no agent.py in {d}")
    jobs = []
    for fen in fens:
        jobs.append((agent_dir, opp_dir, fen, True, args.base_ms, args.increment_ms))
        jobs.append((agent_dir, opp_dir, fen, False, args.base_ms, args.increment_ms))
    started = time.time()
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for r in pool.map(play_one, jobs):
                results.append(r)
                print(
                    _line(len(results), len(jobs), r),
                    flush=True,
                )
    else:
        for job in jobs:
            r = play_one(job)
            results.append(r)
            print(
                _line(len(results), len(jobs), r),
                flush=True,
            )

    wins = sum(1 for r in results if r["score"] == 1.0)
    draws = sum(1 for r in results if r["score"] == 0.5)
    losses = sum(1 for r in results if r["score"] == 0.0)
    n = len(results)
    score = (wins + draws / 2) / n
    terms: dict[str, int] = {}
    for r in results:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    failures = sum(1 for r in results if r["my_failure"])
    depths = [d for r in results for d in r["stats"]["depth"]]
    nodes = [d for r in results for d in r["stats"]["nodes"]]
    times = [d for r in results for d in r["stats"]["time"]]
    pols = [d for r in results for d in r["stats"]["pol"]]
    elo = None
    if 0 < score < 1:
        import math

        elo = -400 * math.log10(1 / score - 1)
    summary = {
        "agent": args.agent,
        "opponent": args.opponent,
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_diff": elo,
        "terminations": terms,
        "agent_failures": failures,
        "avg_depth": sum(depths) / len(depths) if depths else None,
        "avg_nodes": sum(nodes) / len(nodes) if nodes else None,
        "avg_move_time_s": sum(times) / len(times) if times else None,
        "avg_policy_calls": sum(pols) / len(pols) if pols else None,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "minutes": (time.time() - started) / 60,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "games": results}, f, indent=1)


if __name__ == "__main__":
    main()
