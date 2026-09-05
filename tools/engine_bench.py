"""Benchmark every agent in this repository against a real chess engine.

The engine is Stockfish, run locally as the *opponent* and never shipped: the competition
forbids shipping a third-party engine, not measuring yourself against one.  Stockfish plays
with ``UCI_LimitStrength`` at a fixed ``UCI_Elo`` and a fixed think time, so every agent faces
the same calibrated opponent and the scores are comparable across agents and across runs.

Games are played by ``harness.referee.play_match``, the same referee the local arena uses, so
the agent is started through ``harness/runner.py``, gets the real 90 s init budget, is charged
wall time and loses on a flag, an illegal move or a crash exactly as on the platform.  The
engine side is a small object with the same ``start``/``move``/``stop`` interface.

Results are appended to a JSONL file and a re-run skips games already in it, so a run that is
interrupted (or a game count that is raised later) continues rather than starting over.

    python -m tools.engine_bench --games 100 --elo 1320            # every agent
    python -m tools.engine_bench --agents 31 29 --games 20         # a couple of them
    python -m tools.engine_bench --report                          # table from what exists
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.referee import play_match  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402

DEFAULT_RESULTS = REPO / "benchmarks" / "engine_bench.jsonl"
STOCKFISH_CANDIDATES = (
    "agents/21_nnue/training/teacher/stockfish",
    "agents/23_hybrid/training/tools/stockfish",
)


# ---------------------------------------------------------------------------- the engine side


class EngineAgent:
    """Stockfish behind the same interface ``harness.referee`` expects of an agent.

    ``movetime`` is fixed rather than taken from the clock so that the opponent is identical
    in every game of the benchmark, whatever the agent does with its own time.
    """

    def __init__(self, path: Path, elo: int | None, movetime_ms: int, hash_mb: int = 16) -> None:
        self.path = path
        self.elo = elo
        self.movetime_ms = movetime_ms
        self.hash_mb = hash_mb
        self.stderr_tail = ""
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # -- UCI plumbing

    def _send(self, line: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise AgentFailure("crash")
        process.stdin.write(line + "\n")
        process.stdin.flush()

    def _read_until(self, prefix: str, deadline: float) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise AgentFailure("crash")
        while True:
            if time.monotonic() > deadline:
                raise AgentFailure("flag")
            line = process.stdout.readline()
            if not line:
                raise AgentFailure("crash")
            if line.startswith(prefix):
                return line.strip()

    def start(self, init_budget_s: float) -> None:
        self._process = subprocess.Popen(
            [str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + init_budget_s
        self._send("uci")
        self._read_until("uciok", deadline)
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {self.hash_mb}")
        if self.elo is not None:
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {self.elo}")
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok", deadline)

    def move(self, fen: str, time_left_ms: int) -> str:
        # never spend more than a slice of the remaining clock: the referee charges the engine
        # wall time too, and a fixed movetime large enough to matter will flag it in a long
        # game, which hands the agent free wins and measures nothing
        budget = max(20, min(self.movetime_ms, time_left_ms // 30))
        self._send(f"position fen {fen}")
        self._send(f"go movetime {budget}")
        deadline = time.monotonic() + budget / 1000.0 + 30.0
        return self._read_until("bestmove", deadline).split()[1]

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            self._send("quit")
            process.wait(timeout=5)
        except Exception:
            process.kill()
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            self._process = None


def find_stockfish(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"no engine at {path}")
        return path
    for relative in STOCKFISH_CANDIDATES:
        root = REPO / relative
        if root.is_dir():
            for candidate in sorted(root.rglob("*.exe")) + sorted(root.rglob("stockfish*")):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
    raise SystemExit(
        "no Stockfish binary found; pass --engine <path>. It is only ever the opponent: "
        "the competition forbids shipping a third-party engine, not playing against one."
    )


# ---------------------------------------------------------------------------- the agents


@dataclass(frozen=True)
class AgentEntry:
    number: int
    name: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.number:02d}_{self.name}"


def discover_agents() -> list[AgentEntry]:
    """Every directory in the repo that exposes an agent.py the harness can import."""
    found: dict[int, AgentEntry] = {}
    for parent in (REPO / "my-agents", REPO / "agents"):
        if not parent.is_dir():
            continue
        for directory in sorted(parent.iterdir()):
            if not (directory / "agent.py").is_file():
                continue
            head, _, tail = directory.name.partition("_")
            if not head.isdigit():
                continue
            number = int(head)
            found.setdefault(number, AgentEntry(number, tail or directory.name, directory))
    return [found[k] for k in sorted(found)]


# ---------------------------------------------------------------------------- playing


@dataclass(frozen=True)
class Job:
    agent: AgentEntry
    game: int  # 0-based; even = agent plays white
    elo: int
    start_fen: str = chess.STARTING_FEN


def load_openings(path: Path | None) -> list[str]:
    """Opening positions, one FEN per line.

    Games are played in pairs from the same opening with the colours swapped, which removes
    most of the variance caused by the opening itself. Rated games on the platform also start
    from curated positions rather than the initial one, so this is the more representative
    measurement as well as the more sensitive one.
    """
    if path is None:
        return [chess.STARTING_FEN]
    fens = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return fens or [chess.STARTING_FEN]


def game_key(job: Job) -> str:
    return f"{job.agent.label}|elo{job.elo}|g{job.game:03d}"


def plies_of(pgn: str) -> int:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return len(list(game.mainline_moves())) if game is not None else 0


def play_one(job: Job, engine_path: Path, args: argparse.Namespace) -> dict[str, object]:
    plays_white = job.game % 2 == 0
    agent_side = local(job.agent.path)
    engine = EngineAgent(engine_path, job.elo, args.movetime)
    white, black = (agent_side, engine) if plays_white else (engine, agent_side)
    started = time.monotonic()
    try:
        outcome = play_match(  # type: ignore[arg-type]
            white, black, args.base_ms, args.increment_ms, start_fen=job.start_fen
        )
        result, termination, plies = outcome.result, outcome.termination, plies_of(outcome.pgn)
    except Exception as error:  # a broken agent must not stop the benchmark
        result, termination, plies = ("black" if plays_white else "white"), f"error:{error!r}", 0
    # whatever went wrong, the agent's own stderr is the only thing that explains it
    detail = ""
    failed_side = ""
    if termination.startswith(FAILURE_PREFIXES):
        detail = (agent_side.stderr_tail or "")[-1500:]
        # the side that failed is the side that lost the game by failing
        agent_lost = (result == "white") != plays_white
        failed_side = "agent" if agent_lost else "engine"
    if result == "draw" or result == "void":
        points = 0.5
    elif (result == "white") == plays_white:
        points = 1.0
    else:
        points = 0.0
    return {
        "key": game_key(job),
        "agent": job.agent.label,
        "number": job.agent.number,
        "elo": job.elo,
        "game": job.game,
        "agent_white": plays_white,
        "result": result,
        "termination": termination,
        "points": points,
        "plies": plies,
        "seconds": round(time.monotonic() - started, 1),
        "failed_side": failed_side,
        "stderr": detail,
    }


def worker(jobs: Iterator[Job], engine_path: Path, args: argparse.Namespace, out: Path,
           lock: threading.Lock, counter: list[int], total: int) -> None:
    while True:
        with lock:
            job = next(jobs, None)
        if job is None:
            return
        entry = play_one(job, engine_path, args)
        with lock:
            with open(out, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            counter[0] += 1
            print(
                f"[{counter[0]}/{total}] {entry['agent']:<22} g{entry['game']:03d} "
                f"{'W' if entry['agent_white'] else 'B'} -> {entry['points']:.1f} "
                f"({entry['termination']}, {entry['seconds']:.0f}s)",
                flush=True,
            )


# ---------------------------------------------------------------------------- reporting


FAILURE_PREFIXES = ("error", "crash", "illegal", "flag", "init")


def is_failure(entry: dict[str, object]) -> bool:
    """Did this game end by someone failing rather than by a chess result?

    Either side failing makes the game useless as a strength measurement: an engine that
    flags on its own clock hands the agent a free win just as an agent that crashes hands one
    to the engine.
    """
    return str(entry["termination"]).startswith(FAILURE_PREFIXES)


def agent_failed(entry: dict[str, object]) -> bool:
    """Was the *agent* the side that failed? Older records have no field, so fall back."""
    side = entry.get("failed_side")
    if side is not None and side != "":
        return bool(side == "agent")
    return is_failure(entry) and float(entry["points"]) == 0.0  # type: ignore[arg-type]


def load_results(path: Path) -> dict[str, dict[str, object]]:
    """The last line written for a key wins, so a replayed game replaces the earlier one."""
    done: dict[str, dict[str, object]] = {}
    if path.exists():
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    entry = json.loads(line)
                    done[str(entry["key"])] = entry
    return done


def elo_of(score: float, n: int, opponent_elo: int) -> tuple[float, float]:
    """Agent Elo implied by a score against an opponent of known rating, and the 95% half-width."""
    clamped = min(max(score, 0.5 / n), 1 - 0.5 / n)
    elo = opponent_elo - 400 * math.log10(1 / clamped - 1)
    se = math.sqrt(max(clamped * (1 - clamped), 1e-9) / n)
    hi = min(max(clamped + 1.96 * se, 0.5 / n), 1 - 0.5 / n)
    return elo, (opponent_elo - 400 * math.log10(1 / hi - 1)) - elo


def report(
    results: dict[str, dict[str, object]], agents: list[AgentEntry], exclude_failures: bool = False
) -> str:
    """Render the table.

    ``exclude_failures`` scores only games that ended in a chess result.  On a shared machine
    a crash or a flag is usually the machine (see benchmarks/README.md), and counting those as
    losses understates an agent by tens of points; the excluded count stays in the table so the
    reader can see how much was dropped.
    """
    by_elo: dict[int, dict[str, list[dict[str, object]]]] = {}
    for entry in results.values():
        by_elo.setdefault(int(entry["elo"]), {}).setdefault(str(entry["agent"]), []).append(entry)
    lines: list[str] = []
    for elo in sorted(by_elo):
        rows = by_elo[elo]
        scored = "decided games only" if exclude_failures else "all games"
        lines.append(f"\n### vs Stockfish UCI_Elo {elo} ({scored})\n")
        lines.append(
            "| # | agent | games | W | D | L | score | implied Elo | agent failed | "
            "engine failed |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        table = []
        for agent in agents:
            all_games = rows.get(agent.label)
            if not all_games:
                continue
            bad_count = sum(1 for g in all_games if agent_failed(g))
            engine_bad = sum(1 for g in all_games if is_failure(g) and not agent_failed(g))
            games = [g for g in all_games if not is_failure(g)] if exclude_failures else all_games
            if not games:
                continue
            n = len(games)
            wins = sum(1 for g in games if g["points"] == 1.0)
            draws = sum(1 for g in games if g["points"] == 0.5)
            losses = n - wins - draws
            score = sum(float(g["points"]) for g in games) / n
            rating, half = elo_of(score, n, elo)
            table.append((rating, agent, n, wins, draws, losses, score, half, bad_count,
                          engine_bad))
        for rating, agent, n, wins, draws, losses, score, half, bad, ebad in sorted(
            table, key=lambda t: -t[0]
        ):
            lines.append(
                f"| {agent.number} | {agent.name} | {n} | {wins} | {draws} | {losses} | "
                f"{score:.1%} | {rating:.0f} ± {half:.0f} | {bad or ''} | {ebad or ''} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", type=int, nargs="*", default=None, help="agent numbers")
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="NUMBER:LABEL:PATH",
        help="an agent directory outside my-agents/ and agents/, e.g. a variant built to A/B "
        "test a change: '90:dc_base:agents/29_deepchess/variants/base'",
    )
    parser.add_argument("--games", type=int, default=100, help="games per agent per Elo level")
    parser.add_argument("--elo", type=int, nargs="*", default=[1320], help="Stockfish UCI_Elo")
    parser.add_argument("--movetime", type=int, default=100, help="engine ms per move")
    parser.add_argument("--base-ms", type=int, default=10_000, help="agent clock")
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4, help="games in parallel")
    parser.add_argument("--engine", default=None, help="path to the Stockfish binary")
    parser.add_argument(
        "--openings",
        type=Path,
        default=None,
        help="file of opening FENs, one per line; each is played twice, colours swapped",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="play at most this many games this invocation, then exit. Short invocations "
        "survive a machine whose low-memory watchdog kills long-lived background jobs; "
        "nothing is lost either way, since every finished game is appended as it ends",
    )
    parser.add_argument("--report", action="store_true", help="only print the table")
    parser.add_argument(
        "--exclude-failures",
        action="store_true",
        help="score only games that ended in a chess result, and show how many were dropped",
    )
    parser.add_argument(
        "--redo-failures",
        action="store_true",
        help="replay games that ended in a crash/flag/illegal/init, which on a loaded machine "
        "are usually the machine rather than the agent (a replay overwrites the earlier result)",
    )
    parser.add_argument("--report-file", type=Path, default=None, help="also write the table here")
    args = parser.parse_args()

    agents = discover_agents()
    if args.agents:
        wanted = set(args.agents)
        agents = [a for a in agents if a.number in wanted]
    for item in args.extra:
        number, label, path = item.split(":", 2)
        directory = Path(path)
        if not (directory / "agent.py").is_file():
            raise SystemExit(f"no agent.py in {directory}")
        agents.append(AgentEntry(int(number), label, directory.resolve()))
    if not agents:
        raise SystemExit("no agents matched")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_results(args.out)
    if args.redo_failures:
        stale = [key for key, entry in done.items() if is_failure(entry)]
        for key in stale:
            del done[key]
        print(f"replaying {len(stale)} failed games")

    if not args.report:
        engine_path = find_stockfish(args.engine)
        print(f"engine: {engine_path}")
        print(f"{len(agents)} agents, {args.games} games each, Elo levels {args.elo}")
        # round-robin ordering: one game per agent, then the next game.  An interrupted run
        # therefore leaves every agent with a comparable number of games instead of leaving
        # the last agents with none.
        openings = load_openings(args.openings)
        pending = [
            Job(agent, game, elo, openings[(game // 2) % len(openings)])
            for elo in args.elo
            for game in range(args.games)
            for agent in agents
            if game_key(Job(agent, game, elo)) not in done
        ]
        remaining = len(pending)
        if args.max_games > 0:
            pending = pending[: args.max_games]
        print(f"{len(done)} games already played, {remaining} to go, {len(pending)} this run")
        if pending:
            jobs = iter(pending)
            lock = threading.Lock()
            counter = [0]
            threads = [
                threading.Thread(
                    target=worker,
                    args=(jobs, engine_path, args, args.out, lock, counter, len(pending)),
                    daemon=True,
                )
                for _ in range(max(1, args.workers))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        done = load_results(args.out)

    table = report(done, agents, exclude_failures=args.exclude_failures)
    print(table)
    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(table + "\n")
        print(f"\nwritten to {args.report_file}")


if __name__ == "__main__":
    main()
