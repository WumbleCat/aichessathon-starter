"""Build the submission zip for this agent and verify the contest constraints on it.

* builds ``submission.zip`` with ``harness.package`` (agent.py at the root, ``models/`` included)
* checks the unzipped size against the 50 MB cap
* extracts it into a scratch directory and, in a fresh interpreter with only that directory on
  ``sys.path``, measures import time (90 s budget) and plays a few moves through ``get_move``

    python tests/check_submission.py [--out path/to/submission.zip]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
PROJECT = AGENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT))

from harness.package import build  # noqa: E402
from harness.rules import INIT_BUDGET_S, MAX_UNZIPPED_BYTES  # noqa: E402

PROBE = r"""
import sys, time, os
sys.path.insert(0, sys.argv[1])
t0 = time.perf_counter()
import agent
init = time.perf_counter() - t0
import chess
fens = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
]
worst = 0.0
for fen in fens:
    b = chess.Board(fen)
    t0 = time.perf_counter()
    uci = agent.get_move(fen, 2000)
    el = time.perf_counter() - t0
    worst = max(worst, el)
    assert chess.Move.from_uci(uci) in b.legal_moves, (fen, uci)
policy = 'yes' if agent.policy_net is not None else 'NO'
print(f"RESULT init {init:.2f}s worst_move {worst:.2f}s policy {policy}")
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=AGENT_DIR / "submission.zip")
    args = parser.parse_args()

    os.chdir(AGENT_DIR)
    written = build(AGENT_DIR, args.out, ("models",))
    unzipped = sum((AGENT_DIR / name).stat().st_size for name in written)
    print(f"{args.out} ({args.out.stat().st_size:,} bytes zipped, {unzipped:,} unzipped)")
    for name in written:
        print("  " + name)
    assert "agent.py" in written
    assert unzipped < MAX_UNZIPPED_BYTES, (
        f"{unzipped:,} bytes is over the {MAX_UNZIPPED_BYTES:,} cap"
    )
    weights = [n for n in written if n.startswith("models") and n.endswith(".npz")]
    if not weights:
        print("WARNING: no models/*.npz in the zip; the engine will play without the network")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.out) as archive:
            archive.extractall(tmp)
        env = dict(os.environ)
        env.pop("PN_MODEL_PATH", None)
        env.pop("PN_USE_POLICY", None)
        proc = subprocess.run(
            [sys.executable, "-c", PROBE, tmp],
            capture_output=True,
            text=True,
            timeout=INIT_BUDGET_S + 60,
            env=env,
            cwd=tmp,
        )
        tail = proc.stdout.strip().splitlines()
        print("\n".join(line for line in tail if not line.startswith("depth")))
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            raise SystemExit("probe failed")
        result = [line for line in tail if line.startswith("RESULT")][-1]
        init = float(result.split()[2].rstrip("s"))
        assert init < INIT_BUDGET_S, f"import took {init:.1f}s"
    print("submission check OK")


if __name__ == "__main__":
    main()
