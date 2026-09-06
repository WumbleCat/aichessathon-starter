"""Check a benchmark results file for the mistakes that would quietly invalidate a table.

Run it before trusting a report. Every check here corresponds to a bug that actually happened
while building these results, or to one that would be invisible in the summary table.

    python -m tools.verify_bench benchmarks/all32_2600.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DECISIVE = {"checkmate", "stalemate", "threefold_repetition", "insufficient_material",
            "fifty_moves", "adjudication", "cap", "variant_win", "variant_loss", "variant_draw"}
FAILURES = ("error", "crash", "illegal", "flag", "init")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expect-games", type=int, default=0, help="games per agent per level")
    args = parser.parse_args()

    lines = args.path.read_text(encoding="utf-8").splitlines()
    raw = [json.loads(line) for line in lines if line.strip()]
    by_key: dict[str, list[dict]] = collections.defaultdict(list)
    for row in raw:
        by_key[row["key"]].append(row)
    print(f"{len(raw)} lines, {len(by_key)} distinct games")

    problems = 0

    # 1. duplicates that disagree: two runs played the same game and got different results,
    #    which means the two runs were not the same experiment (different model, flags, build)
    conflicting = 0
    for key, rows in by_key.items():
        if len(rows) > 1 and len({r["points"] for r in rows}) > 1:
            conflicting += 1
            if conflicting <= 3:
                print(f"  conflicting duplicate {key}: points {[r['points'] for r in rows]}")
    if conflicting:
        problems += 1
        print(f"! {conflicting} games were replayed with a DIFFERENT result. The file mixes "
              f"runs that were not the same experiment; the last line wins, which may be wrong.")
    else:
        dups = sum(len(r) - 1 for r in by_key.values())
        print(f"duplicates: {dups} (all agreeing)" if dups else "duplicates: none")

    # 2. colour balance: an unbalanced set is a biased set, since White scores better
    for agent, rows in group(by_key, "agent").items():
        whites = sum(1 for r in rows if r["agent_white"])
        if abs(whites - (len(rows) - whites)) > 1:
            problems += 1
            print(f"! {agent}: {whites} games as White, {len(rows) - whites} as Black")

    # 3. failures, split by which side caused them
    agent_fail = [r for r in by_key.values() for r in [r[-1]]
                  if str(r["termination"]).startswith(FAILURES) and r.get("failed_side") == "agent"]
    engine_fail = [r for r in by_key.values() for r in [r[-1]]
                   if str(r["termination"]).startswith(FAILURES)
                   and r.get("failed_side") == "engine"]
    print(f"failures: {len(agent_fail)} by an agent, {len(engine_fail)} by the engine")
    if engine_fail:
        problems += 1
        print("! the engine failed in some games; those are free points for the agent and the "
              "affected rows are not a strength measurement")

    # 4. terminations that are not a real chess result and not a known failure
    unknown = {str(r[-1]["termination"]) for r in by_key.values()
               if str(r[-1]["termination"]) not in DECISIVE
               and not str(r[-1]["termination"]).startswith(FAILURES)}
    if unknown:
        problems += 1
        print(f"! unrecognised terminations: {sorted(unknown)}")

    # 5. every agent should have the same settings; a mixed elo in one file is a mixed table
    for field in ("elo",):
        values = {r[-1][field] for r in by_key.values()}
        print(f"{field} values present: {sorted(values)}")

    # 6. completeness
    counts = {a: len(rows) for a, rows in group(by_key, "agent").items()}
    if args.expect_games:
        short = {a: n for a, n in counts.items() if n < args.expect_games}
        if short:
            print(f"incomplete ({args.expect_games} expected): "
                  + ", ".join(f"{a} {n}" for a, n in sorted(short.items())))
    print(f"agents: {len(counts)}, games per agent min {min(counts.values())} "
          f"max {max(counts.values())}")

    print("\nOK: no integrity problems found" if not problems
          else f"\n{problems} problem(s) above")
    sys.exit(1 if problems else 0)


def group(by_key: dict[str, list[dict]], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for rows in by_key.values():
        out[str(rows[-1][field])].append(rows[-1])
    return out


if __name__ == "__main__":
    main()
