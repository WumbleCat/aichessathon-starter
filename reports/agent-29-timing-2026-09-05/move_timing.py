"""Per-move wall time of agents/29_deepchess through the unmodified harness referee.

Plays paired games (both colours) against a baseline at a given time control, timing every
move request exactly as the referee does (time.monotonic around Agent.move), and reports
median / p90 / p99 / max move time and the largest fraction of the remaining clock used.
Nothing under agents/29_deepchess is written; per-move records go to --out as JSONL.

    .venv/Scripts/python.exe move_timing.py --opponent baselines/greedy --games 2 \
        --base-ms 120000 --increment-ms 500 --out move_times.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("E:/sourcecode/ai-chess-original/aichessathon-starter")
sys.path.insert(0, str(ROOT))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.sandbox import Agent, local  # noqa: E402


class TimedAgent(Agent):
    """The harness agent with a record of every move request."""

    def __init__(self, command: list[str], label: str, records: list[dict]) -> None:
        super().__init__(command)
        self.label = label
        self.records = records
        self.init_s: float | None = None

    def start(self, init_budget_s: float) -> None:
        t0 = time.monotonic()
        super().start(init_budget_s)
        self.init_s = time.monotonic() - t0

    def move(self, fen: str, time_left_ms: int) -> str:
        t0 = time.monotonic()
        try:
            uci = super().move(fen, time_left_ms)
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self.records.append({
                "agent": self.label, "fen": fen, "time_left_ms": time_left_ms,
                "move_ms": round(elapsed_ms, 1),
                "clock_fraction": round(elapsed_ms / max(time_left_ms, 1), 4),
            })
        return uci


def timed(directory: Path, label: str, records: list[dict]) -> TimedAgent:
    proto = local(directory)
    return TimedAgent(proto.command, label, records)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(p / 100.0 * (len(ordered) - 1))))
    return ordered[k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=ROOT / "agents/29_deepchess")
    parser.add_argument("--opponent", type=Path, default=ROOT / "baselines/greedy")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict] = []
    inits: list[float] = []
    results: list[tuple[str, str, bool]] = []
    for game in range(args.games):
        plays_white = game % 2 == 0
        ours = timed(args.agent.resolve(), "agent29", records)
        theirs = timed(args.opponent.resolve(), "opponent", records)
        white, black = (ours, theirs) if plays_white else (theirs, ours)
        outcome = play_match(white, black, args.base_ms, args.increment_ms)
        if ours.init_s is not None:
            inits.append(ours.init_s)
        results.append((outcome.result, outcome.termination, plays_white))
        init = f"{ours.init_s:.1f}s" if ours.init_s is not None else "n/a"
        print(f"game {game + 1}/{args.games}: {outcome.result} by {outcome.termination} "
              f"(agent29 {'white' if plays_white else 'black'}, init {init})", flush=True)
        if outcome.termination in FAILED_TERMINATIONS:
            print("stderr tail:", ours.stderr_tail[-600:], flush=True)

    with args.out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    ours_ms = [r["move_ms"] for r in records if r["agent"] == "agent29"]
    fractions = [r["clock_fraction"] for r in records if r["agent"] == "agent29"]
    wins = sum(1 for res, _, w in results if res in ("white", "black") and (res == "white") == w)
    draws = sum(1 for res, _, _ in results if res in ("draw", "void"))
    losses = len(results) - wins - draws
    summary = {
        "agent": str(args.agent), "opponent": str(args.opponent), "games": args.games,
        "base_ms": args.base_ms, "increment_ms": args.increment_ms,
        "score": f"+{wins} ={draws} -{losses}",
        "terminations": {t: sum(1 for _, tt, _ in results if tt == t)
                         for t in {tt for _, tt, _ in results}},
        "agent29_moves": len(ours_ms),
        "init_s": [round(x, 1) for x in inits],
        "move_ms": {
            "median": round(statistics.median(ours_ms), 1) if ours_ms else None,
            "mean": round(statistics.fmean(ours_ms), 1) if ours_ms else None,
            "p90": round(percentile(ours_ms, 90), 1),
            "p99": round(percentile(ours_ms, 99), 1),
            "max": round(max(ours_ms), 1) if ours_ms else None,
        },
        "max_clock_fraction": round(max(fractions), 4) if fractions else None,
    }
    print(json.dumps(summary, indent=1))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
