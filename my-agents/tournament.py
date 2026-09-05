"""Round-robin tournament between every bot in my-agents/.

Every unordered pair of bots plays --games games (colours alternating), games run in
parallel worker processes, and every finished game is appended to games.jsonl so the run
can be interrupted and resumed. At the end (or with --report-only) a ranking is written
to RANKING.md and results.json in the output directory.

    .venv/Scripts/python.exe my-agents/tournament.py --games 100 --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.rules import PLY_CAP  # noqa: E402
from harness.sandbox import local  # noqa: E402

AGENTS_DIR = ROOT / "my-agents"
OUT_DIR = AGENTS_DIR / "results" / "tournament"
FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def discover_bots() -> list[str]:
    return sorted(p.name for p in AGENTS_DIR.iterdir() if p.is_dir() and (p / "agent.py").is_file())


def play_one(a: str, b: str, index: int, base_ms: int, increment_ms: int, ply_cap: int) -> dict:
    white, black = (a, b) if index % 2 == 0 else (b, a)
    started = time.monotonic()
    outcome = play_match(
        local(AGENTS_DIR / white),
        local(AGENTS_DIR / black),
        base_ms,
        increment_ms,
        ply_cap=ply_cap,
    )
    return {
        "a": a,
        "b": b,
        "index": index,
        "white": white,
        "black": black,
        "result": outcome.result,
        "termination": outcome.termination,
        "seconds": round(time.monotonic() - started, 1),
        "pgn": outcome.pgn,
    }


def load_games(path: Path) -> list[dict]:
    if not path.exists():
        return []
    games = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            games.append(json.loads(line))
    return games


def run(
    bots: list[str],
    games_per_pair: int,
    workers: int,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
    out_dir: Path,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / "games.jsonl"
    done = load_games(games_path)
    done_keys = {(g["a"], g["b"], g["index"]) for g in done}
    pairs = list(combinations(bots, 2))
    todo = [(a, b, i) for a, b in pairs for i in range(games_per_pair) if (a, b, i) not in done_keys]
    total = len(pairs) * games_per_pair
    print(
        f"{len(bots)} bots, {total} games total, {len(done)} already done, "
        f"{len(todo)} to play, {workers} workers",
        flush=True,
    )
    started = time.monotonic()
    finished = 0
    with games_path.open("a", encoding="utf-8") as sink, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(play_one, a, b, i, base_ms, increment_ms, ply_cap): (a, b, i) for a, b, i in todo
        }
        for future in as_completed(futures):
            a, b, i = futures[future]
            try:
                game = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"ERROR {a} vs {b} #{i}: {error!r}", flush=True)
                continue
            sink.write(json.dumps(game) + "\n")
            sink.flush()
            done.append(game)
            finished += 1
            elapsed = time.monotonic() - started
            rate = finished / elapsed if elapsed else 0
            remaining = (len(todo) - finished) / rate if rate else float("inf")
            print(
                f"[{len(done)}/{total}] {game['white']} vs {game['black']}: {game['result']} by "
                f"{game['termination']} ({game['seconds']}s) | elapsed {elapsed / 60:.0f}m, "
                f"eta {remaining / 60:.0f}m",
                flush=True,
            )
    return done


def report(bots: list[str], games: list[dict], games_per_pair: int, out_dir: Path) -> str:
    keys = ("wins", "draws", "losses", "games", "white_wins", "black_wins", "flags", "failures")
    stats = {b: dict.fromkeys(keys, 0) for b in bots}
    h2h = {a: {b: {"w": 0, "d": 0, "l": 0} for b in bots} for a in bots}
    terminations: dict[str, int] = {}
    for g in games:
        w, bl, r = g["white"], g["black"], g["result"]
        if w not in stats or bl not in stats:
            continue
        terminations[g["termination"]] = terminations.get(g["termination"], 0) + 1
        stats[w]["games"] += 1
        stats[bl]["games"] += 1
        if r in ("draw", "void"):
            stats[w]["draws"] += 1
            stats[bl]["draws"] += 1
            h2h[w][bl]["d"] += 1
            h2h[bl][w]["d"] += 1
            continue
        winner, loser = (w, bl) if r == "white" else (bl, w)
        stats[winner]["wins"] += 1
        stats[loser]["losses"] += 1
        h2h[winner][loser]["w"] += 1
        h2h[loser][winner]["l"] += 1
        stats[winner]["white_wins" if r == "white" else "black_wins"] += 1
        if g["termination"] == "flag":
            stats[loser]["flags"] += 1
        if g["termination"] in FAILED_TERMINATIONS:
            stats[loser]["failures"] += 1

    def win_rate(b: str) -> float:
        s = stats[b]
        return s["wins"] / s["games"] if s["games"] else 0.0

    def score(b: str) -> float:
        s = stats[b]
        return (s["wins"] + s["draws"] / 2) / s["games"] if s["games"] else 0.0

    ranking = sorted(bots, key=lambda b: (win_rate(b), score(b)), reverse=True)
    lines = [
        "# Round-robin tournament ranking",
        "",
        f"Every bot played every other bot {games_per_pair} times (colours alternating), "
        f"{len(games)} games in total, at the arena's fast time control (10 s + 0.1 s per side). "
        "Ranked by win rate (wins / games); score counts a draw as half a win.",
        "",
        "| Rank | Bot | Win rate | Score | W | D | L | Games | Wins as White | Wins as Black | Losses on time |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, b in enumerate(ranking, 1):
        s = stats[b]
        lines.append(
            f"| {rank} | `{b}` | **{win_rate(b):.1%}** | {score(b):.1%} | {s['wins']} | {s['draws']} | "
            f"{s['losses']} | {s['games']} | {s['white_wins']} | {s['black_wins']} | {s['flags']} |"
        )
    lines += ["", "## Head-to-head (row bot's W-D-L against column bot)", ""]
    short = {b: b.split("_")[0] for b in bots}
    lines.append("| | " + " | ".join(short[b] for b in ranking) + " |")
    lines.append("|---|" + "---|" * len(ranking))
    for a in ranking:
        cells = []
        for b in ranking:
            if a == b:
                cells.append("-")
            else:
                c = h2h[a][b]
                cells.append(f"{c['w']}-{c['d']}-{c['l']}")
        lines.append(f"| `{a}` | " + " | ".join(cells) + " |")
    lines += ["", "Column keys: " + ", ".join(f"{short[b]} = `{b}`" for b in ranking), ""]
    lines += ["## Head-to-head win rate (row bot's wins / games played against column bot)", ""]
    lines.append("| | " + " | ".join(short[b] for b in ranking) + " |")
    lines.append("|---|" + "---|" * len(ranking))
    for a in ranking:
        cells = []
        for b in ranking:
            c = h2h[a][b]
            played = c["w"] + c["d"] + c["l"]
            if a == b or played == 0:
                cells.append("-")
            else:
                cells.append(f"{c['w'] / played:.0%}")
        lines.append(f"| `{a}` | " + " | ".join(cells) + " |")
    lines += ["", "## Terminations", ""]
    lines += [f"- {name}: {count}" for name, count in sorted(terminations.items(), key=lambda kv: -kv[1])]
    text = "\n".join(lines) + "\n"
    (out_dir / "RANKING.md").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(
            {"ranking": ranking, "stats": stats, "head_to_head": h2h, "terminations": terminations},
            indent=2,
        ),
        encoding="utf-8",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="games per pairing")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--bots", nargs="*", help="restrict to these bot directory names")
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()

    bots = arguments.bots or discover_bots()
    if arguments.report_only:
        games = load_games(arguments.out / "games.jsonl")
    else:
        games = run(
            bots,
            arguments.games,
            arguments.workers,
            arguments.base_ms,
            arguments.increment_ms,
            arguments.ply_cap,
            arguments.out,
        )
    print(report(bots, games, arguments.games, arguments.out))


if __name__ == "__main__":
    main()
