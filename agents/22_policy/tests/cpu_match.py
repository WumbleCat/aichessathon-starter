"""Paired in-process match between two engine configurations on a CPU-time clock.

The development machine is shared with many other jobs, so wall-clock arenas mostly measure the
load. Here both sides search with ``time.process_time`` as the clock, which charges the policy
side for its inference cost but ignores everything the rest of the machine is doing. Every
opening is played twice with colours swapped.

    python tests/cpu_match.py --a policy --b nopolicy --pairs 10 --budget 0.35 --out results/x.json

Configurations: ``nopolicy``, ``policy`` (root + depth >= 4, the shipped default), ``rootonly``,
``d2``, ``d3``, ``d5``, ``nolmr`` (prior for ordering but not for LMR), ``searchless``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import chess  # noqa: E402
from paired_arena import openings  # noqa: E402
from pn_eval import PIECE_VALUE  # noqa: E402
from pn_policy import load_policy  # noqa: E402
from pn_search import Searcher  # noqa: E402

PLY_CAP = 300
MAX_DEPTH = 40


class Player:
    def __init__(self, name: str, net: object | None) -> None:
        self.name = name
        self.searchless = name == "searchless"
        self.net = net
        prior = None if net is None or name == "nopolicy" else net.prior  # type: ignore[attr-defined]
        min_depth = {"rootonly": 99, "d2": 2, "d3": 3, "d5": 5}.get(name, 4)
        self.searcher = Searcher(
            prior=prior,
            policy_min_depth=min_depth,
            policy_root=True,
            policy_lmr=name != "nolmr",
            clock=time.process_time,
        )
        self.depths: list[int] = []
        self.nodes: list[int] = []
        self.calls: list[int] = []
        self.cpu: list[float] = []

    def move(self, board: chess.Board, budget: float) -> chess.Move:
        legal = list(board.legal_moves)
        if self.searchless:
            p = self.net.prior(board)  # type: ignore[attr-defined]
            return max(legal, key=lambda m: p.get(m, 0.0))
        self.searcher.remember_position(board)
        c0 = time.process_time()
        result = self.searcher.search(board, MAX_DEPTH, time_budget=budget)
        self.cpu.append(time.process_time() - c0)
        self.depths.append(result.depth)
        self.nodes.append(result.stats.nodes)
        self.calls.append(result.stats.policy_calls)
        move = result.move
        return move if move is not None and move in legal else legal[0]

    def summary(self) -> dict:
        n = max(len(self.depths), 1)
        return {
            "name": self.name,
            "moves": len(self.depths),
            "avg_depth": sum(self.depths) / n,
            "avg_nodes": sum(self.nodes) / n,
            "avg_policy_calls": sum(self.calls) / n,
            "avg_cpu_s": sum(self.cpu) / n,
        }


def material(board: chess.Board) -> int:
    total = 0
    for piece, value in PIECE_VALUE.items():
        if piece == chess.KING:
            continue
        total += value * (len(board.pieces(piece, True)) - len(board.pieces(piece, False)))
    return total


def play(white: Player, black: Player, fen: str, budget: float) -> tuple[float, str]:
    """Return (white score, termination)."""
    board = chess.Board(fen)
    white.searcher.new_game()
    black.searcher.new_game()
    plies = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.5, outcome.termination.name.lower()
            return (1.0 if outcome.winner else 0.0), outcome.termination.name.lower()
        if plies >= PLY_CAP:
            m = material(board)
            if abs(m) < 200:
                return 0.5, "ply_cap_draw"
            return (1.0 if m > 0 else 0.0), "ply_cap_material"
        mover = white if board.turn else black
        board.push(mover.move(board, budget))
        plies += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="policy")
    parser.add_argument("--b", default="nopolicy")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--budget", type=float, default=0.35, help="CPU seconds per move")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model", default=os.path.join(ROOT, "models", "policy.npz"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    net = load_policy(args.model, prefer="numpy")
    if net is None:
        raise SystemExit("no model at " + args.model)
    a, b = Player(args.a, net), Player(args.b, net)
    fens = openings(args.pairs, args.seed)
    games = []
    score_a = 0.0
    started = time.time()
    for fen in fens:
        for a_white in (True, False):
            white, black = (a, b) if a_white else (b, a)
            ws, term = play(white, black, fen, args.budget)
            sa = ws if a_white else 1.0 - ws
            score_a += sa
            games.append({"fen": fen, "a_white": a_white, "score_a": sa, "termination": term})
            n = len(games)
            print(
                f"game {n}/{2 * len(fens)}: {args.a} {'white' if a_white else 'black'} "
                f"score {sa} by {term}  running {score_a}/{n} ({time.time() - started:.0f}s)",
                flush=True,
            )
    n = len(games)
    wins = sum(1 for g in games if g["score_a"] == 1.0)
    draws = sum(1 for g in games if g["score_a"] == 0.5)
    losses = n - wins - draws
    p = score_a / n
    elo = None if p in (0.0, 1.0) else -400 * __import__("math").log10(1 / p - 1)
    summary = {
        "a": a.summary(),
        "b": b.summary(),
        "games": n,
        "a_wins": wins,
        "draws": draws,
        "a_losses": losses,
        "a_score": p,
        "a_elo_diff": elo,
        "budget_cpu_s": args.budget,
        "seed": args.seed,
        "minutes": (time.time() - started) / 60,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "games": games}, f, indent=1)


if __name__ == "__main__":
    main()
