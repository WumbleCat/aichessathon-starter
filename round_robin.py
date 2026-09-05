"""Full round robin between every agent in `agents/`, with a resumable score table.

Every ordered pair of agents plays `--games` games with alternating colours, each game in
a pair of fresh agent processes exactly as `harness.referee` plays one. Results are appended
to a JSONL file, so the run can be stopped and restarted without losing games.

    python round_robin.py --games 100 --workers 6      # play
    python round_robin.py --report                     # standings from whatever has finished

The one deviation from `harness.arena` is that the two agent processes for a game start
concurrently instead of one after the other. That mirrors the platform, where each side runs
in its own container, and it matters here because some agents spend 60 s compiling at import.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.referee import play_match
from harness.rules import INIT_BUDGET_S, PLY_CAP
from harness.sandbox import Agent, AgentFailure

REPO = Path(__file__).resolve().parent
AGENT_ROOT = REPO / "agents"
RESULT_ROOT = REPO / "round_robin_results"
RUNNER = REPO / "harness" / "runner.py"

BASE_MS = 10_000
INCREMENT_MS = 100


class WarmAgent(Agent):
    """An agent whose process can be started before the referee asks for it.

    `warm()` pays the import cost up front and remembers a failure; the referee's own
    `start()` then either returns at once or re-raises that failure, so the game is scored
    exactly as it would be by `harness.referee`.
    """

    def __init__(self, command: list[str]) -> None:
        super().__init__(command)
        self._warmed = False
        self._warm_failure: AgentFailure | None = None

    def warm(self, init_budget_s: float) -> None:
        try:
            super().start(init_budget_s)
        except AgentFailure as failure:
            self._warm_failure = failure
        self._warmed = True

    def start(self, init_budget_s: float) -> None:
        if not self._warmed:
            super().start(init_budget_s)
            return
        if self._warm_failure is not None:
            raise self._warm_failure


@dataclass(frozen=True)
class Task:
    """One game: `left` and `right` name the pair, `game` fixes the colours."""

    left: str
    right: str
    game: int

    @property
    def white(self) -> str:
        return self.left if self.game % 2 == 0 else self.right

    @property
    def black(self) -> str:
        return self.right if self.game % 2 == 0 else self.left

    @property
    def key(self) -> str:
        return f"{self.left}|{self.right}|{self.game}"


def agent_names() -> list[str]:
    return sorted(d.name for d in AGENT_ROOT.iterdir() if (d / "agent.py").exists())


def play_one(payload: tuple[str, str, int, int, int, int]) -> dict[str, Any]:
    """Worker entry point: play a single game in two fresh processes."""
    left, right, game, base_ms, increment_ms, ply_cap = payload
    task = Task(left, right, game)
    started = time.monotonic()
    white = WarmAgent([sys.executable, str(RUNNER), str(AGENT_ROOT / task.white)])
    black = WarmAgent([sys.executable, str(RUNNER), str(AGENT_ROOT / task.black)])
    warmers = [
        threading.Thread(target=white.warm, args=(INIT_BUDGET_S,)),
        threading.Thread(target=black.warm, args=(INIT_BUDGET_S,)),
    ]
    for warmer in warmers:
        warmer.start()
    for warmer in warmers:
        warmer.join()
    try:
        outcome = play_match(white, black, base_ms, increment_ms, ply_cap=ply_cap)
        result, termination = outcome.result, outcome.termination
    except Exception as error:  # a harness fault must not take the pool down
        result, termination = "void", f"harness_error:{type(error).__name__}"
    return {
        "left": left,
        "right": right,
        "game": game,
        "white": task.white,
        "result": result,
        "termination": termination,
        "seconds": round(time.monotonic() - started, 1),
    }


def schedule(names: list[str], games: int) -> list[Task]:
    """Round major: one game of every pairing, then the next, so early standings are complete."""
    pairs = list(itertools.combinations(names, 2))
    return [Task(left, right, game) for game in range(games) for left, right in pairs]


def finished(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a hard stop
            done.add(f"{record['left']}|{record['right']}|{record['game']}")
    return done


def load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def run(
    names: list[str], games: int, workers: int, path: Path, arguments: argparse.Namespace
) -> None:
    done = finished(path)
    todo = [task for task in schedule(names, games) if task.key not in done]
    total = len(names) * (len(names) - 1) // 2 * games
    print(f"{len(names)} agents, {total} games, {len(done)} already played, {len(todo)} to go")
    print(f"time control {arguments.base_ms} ms + {arguments.increment_ms} ms, {workers} workers")
    sys.stdout.flush()
    if not todo:
        return

    started = time.monotonic()
    played = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink, ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future[dict[str, Any]], Task] = {}
        queue = iter(todo)

        def submit_more() -> None:
            while len(pending) < workers * 2:
                try:
                    task = next(queue)
                except StopIteration:
                    return
                payload = (
                    task.left,
                    task.right,
                    task.game,
                    arguments.base_ms,
                    arguments.increment_ms,
                    arguments.ply_cap,
                )
                pending[pool.submit(play_one, payload)] = task

        submit_more()
        while pending:
            for future in as_completed(list(pending)):
                pending.pop(future, None)
                record = future.result()
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                played += 1
                rate = played / max(time.monotonic() - started, 1e-9)
                left = len(todo) - played
                eta_h = left / rate / 3600 if rate > 0 else float("inf")
                print(
                    f"[{played}/{len(todo)}] {record['left']} vs {record['right']}"
                    f" #{record['game']} -> {record['result']} by {record['termination']}"
                    f" ({record['seconds']}s) | {rate * 3600:.0f} games/h, eta {eta_h:.1f}h"
                )
                sys.stdout.flush()
                submit_more()
                break


def bradley_terry(records: list[dict[str, Any]], names: list[str]) -> dict[str, float]:
    """Fit an Elo rating per agent by maximum likelihood, anchored to a mean of zero."""
    index = {name: i for i, name in enumerate(names)}
    scale = math.log(10.0) / 400.0
    # points[i][j] = points agent i took from agent j, games[i][j] = games between them
    points = [[0.0] * len(names) for _ in names]
    counts = [[0.0] * len(names) for _ in names]
    for record in records:
        i, j = index[record["left"]], index[record["right"]]
        share = score_for(record, record["left"])
        points[i][j] += share
        points[j][i] += 1.0 - share
        counts[i][j] += 1.0
        counts[j][i] += 1.0
    # a half point each against a phantom 0-rated opponent keeps a perfect record finite
    prior = 2.0
    rating = [0.0] * len(names)
    for _ in range(500):
        step = 0.0
        for i in range(len(names)):
            # the step is Newton's: residual over curvature, so it does not depend on how many
            # games have been played and stays stable from a handful up to thousands
            chance = expected(rating[i], 0.0, scale)
            residual = prior * (0.5 - chance)
            curvature = prior * chance * (1.0 - chance)
            for j in range(len(names)):
                if counts[i][j] == 0.0:
                    continue
                chance = expected(rating[i], rating[j], scale)
                residual += points[i][j] - counts[i][j] * chance
                curvature += counts[i][j] * chance * (1.0 - chance)
            move = residual / (scale * max(curvature, 1e-12))
            move = max(-200.0, min(200.0, move))  # a cap keeps the first steps from overshooting
            rating[i] += move
            step = max(step, abs(move))
        mean = sum(rating) / len(rating)
        rating = [value - mean for value in rating]
        if step < 1e-6:
            break
    return dict(zip(names, rating, strict=True))


def expected(mine: float, theirs: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-(mine - theirs) * scale))


def score_for(record: dict[str, Any], name: str) -> float:
    """Points `name` took from this game. A void game is half a point to each side."""
    if record["result"] in ("draw", "void"):
        return 0.5
    won_as_white = record["result"] == "white"
    was_white = record["white"] == name
    return 1.0 if won_as_white == was_white else 0.0


def report(path: Path, names: list[str]) -> None:
    records = [r for r in load(path) if r["left"] in names and r["right"] in names]
    if not records:
        print("no games played yet")
        return

    wins: dict[str, int] = dict.fromkeys(names, 0)
    draws: dict[str, int] = dict.fromkeys(names, 0)
    losses: dict[str, int] = dict.fromkeys(names, 0)
    faults: dict[str, int] = dict.fromkeys(names, 0)
    cross: dict[tuple[str, str], list[float]] = {}
    terminations: dict[str, int] = {}

    for record in records:
        terminations[record["termination"]] = terminations.get(record["termination"], 0) + 1
        for name, other in ((record["left"], record["right"]), (record["right"], record["left"])):
            share = score_for(record, name)
            if share == 1.0:
                wins[name] += 1
            elif share == 0.5:
                draws[name] += 1
            else:
                losses[name] += 1
            cross.setdefault((name, other), []).append(share)
        if record["termination"] in ("crash", "illegal", "flag", "init"):
            loser = record["black"] if record["result"] == "white" else record["white"]
            faults[loser] += 1

    ratings = bradley_terry(records, names)
    played = {name: wins[name] + draws[name] + losses[name] for name in names}
    scored = {name: wins[name] + draws[name] / 2 for name in names}
    order = sorted(names, key=lambda n: (-(scored[n] / max(played[n], 1)), n))

    print(f"\n{len(records)} games played\n")
    header = (
        f"{'#':>3} {'agent':22} {'games':>6} {'W':>5} {'D':>5} {'L':>5}"
        f" {'score':>7} {'elo':>7} {'faults':>7}"
    )
    print(header)
    print("-" * len(header))
    for rank, name in enumerate(order, 1):
        rate = scored[name] / max(played[name], 1)
        print(
            f"{rank:>3} {name:22} {played[name]:>6} {wins[name]:>5} {draws[name]:>5}"
            f" {losses[name]:>5} {rate:>6.1%} {ratings[name]:>+7.0f} {faults[name]:>7}"
        )

    print("\ncross table, row's score against column, in percent")
    print("    " + "".join(f"{name.split('_')[0]:>6}" for name in order))
    for name in order:
        cells = []
        for other in order:
            shares = cross.get((name, other))
            cells.append("     ." if not shares else f"{100 * sum(shares) / len(shares):>6.0f}")
        print(f"{name.split('_')[0]:>4}" + "".join(cells) + f"  {name}")

    ranked = sorted(terminations.items(), key=lambda pair: -pair[1])
    print("\nterminations: " + ", ".join(f"{name} {count}" for name, count in ranked))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="games per pairing")
    parser.add_argument("--workers", type=int, default=6, help="games played at once")
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--results", type=Path, default=RESULT_ROOT / "games.jsonl")
    parser.add_argument("--report", action="store_true", help="print standings and exit")
    parser.add_argument("--only", default="", help="comma separated agent names, for a smoke test")
    arguments = parser.parse_args()

    os.chdir(REPO)
    names = agent_names()
    if arguments.only:
        wanted = {part.strip() for part in arguments.only.split(",") if part.strip()}
        missing = wanted - set(names)
        if missing:
            raise SystemExit("unknown agents: " + ", ".join(sorted(missing)))
        names = [name for name in names if name in wanted]
    if arguments.report:
        report(arguments.results, names)
        return
    run(names, arguments.games, arguments.workers, arguments.results, arguments)
    report(arguments.results, names)


if __name__ == "__main__":
    main()
