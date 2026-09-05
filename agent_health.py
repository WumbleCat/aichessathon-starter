"""Check what an agent actually does when it plays, on the machine as it is right now.

Tournament results only say who won. This says whether an agent is playing the engine it was
written to play: how long it takes to import, how much of its clock it spends per move, and what
it wrote to stderr. Several agents in `agents/` compile a numba search in a background thread and
fall back to pure Python until it lands, and on a loaded machine the compile may never land.

    python agent_health.py                       # every agent, one opening each
    python agent_health.py --only 20_pvs         # just one
    python agent_health.py --idle 75             # leave it alone first, then play

`--idle` is the interesting one. A compile thread that finishes during an idle wait but never
during play is being starved of the GIL by the agent's own Python fallback, which is the agent's
bug. One that finishes in neither case is short of processor, which is the machine's fault.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import chess

from harness.rules import INIT_BUDGET_S
from harness.sandbox import AgentFailure, local

REPO = Path(__file__).resolve().parent
AGENT_ROOT = REPO / "agents"

# a normal opening, so every agent is asked the same questions
OPENING = ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4", "d2d4", "e4d6")
BASE_MS = 10_000
INCREMENT_MS = 100


@dataclass
class Health:
    agent: str
    init_s: float
    move_times: list[float]
    stderr: str
    failure: str | None = None

    @property
    def median_move_s(self) -> float:
        if not self.move_times:
            return 0.0
        ordered = sorted(self.move_times)
        return ordered[len(ordered) // 2]


def inspect(name: str, idle_s: float, moves: int) -> Health:
    agent = local(AGENT_ROOT / name)
    began = time.monotonic()
    init_s = 0.0
    times: list[float] = []
    failure: str | None = None
    try:
        agent.start(INIT_BUDGET_S)
        init_s = time.monotonic() - began
        if idle_s > 0:
            time.sleep(idle_s)  # the agent's background threads have the machine to themselves
        board = chess.Board()
        clock = float(BASE_MS)
        for uci in OPENING[:moves]:
            if board.is_game_over():
                break
            started = time.monotonic()
            agent.move(board.fen(), int(clock))
            spent = (time.monotonic() - started) * 1000.0
            clock = clock - spent + INCREMENT_MS
            times.append(round(spent / 1000.0, 2))
            board.push(chess.Move.from_uci(uci))
    except AgentFailure as error:
        failure = error.reason
    finally:
        agent.stop()
    return Health(name, init_s, times, agent.stderr_tail.strip(), failure)


def show(health: Health) -> None:
    print(f"\n=== {health.agent} ===")
    if health.failure is not None:
        print(f"  FAILED: {health.failure} after {health.init_s:.1f}s")
    else:
        print(f"  import {health.init_s:.1f}s of the {INIT_BUDGET_S:.0f}s budget")
        print(f"  move times {health.move_times}, median {health.median_move_s:.2f}s")
    if health.stderr:
        print("  stderr:")
        for line in health.stderr.splitlines()[-8:]:
            print("    " + line)
    else:
        print("  stderr: silent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma separated agent names")
    parser.add_argument("--idle", type=float, default=0.0, help="seconds to wait before playing")
    parser.add_argument("--moves", type=int, default=len(OPENING))
    arguments = parser.parse_args()

    names = sorted(d.name for d in AGENT_ROOT.iterdir() if (d / "agent.py").exists())
    if arguments.only:
        wanted = {part.strip() for part in arguments.only.split(",") if part.strip()}
        missing = wanted - set(names)
        if missing:
            raise SystemExit("unknown agents: " + ", ".join(sorted(missing)))
        names = [name for name in names if name in wanted]

    for name in names:
        show(inspect(name, arguments.idle, arguments.moves))


if __name__ == "__main__":
    main()
