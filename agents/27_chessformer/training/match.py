"""Paired, node-budgeted self-play match: engine A vs engine B, same openings, both colours.

Wall-clock results on the shared dev box are dominated by machine load, so this match runs
both sides in one process at a fixed node budget per move. A network call is charged
--policy-cost nodes (default: measured as model latency x engine nodes/s at start-up), so
the price of consulting the network is part of the budget and the comparison is fair.

    python training/match.py --a models/chessformer.npz --b none --pairs 20 --nodes 4000

Prints W/D/L for A, an Elo estimate, and the average node / network-call counts.
"""

import argparse
import math
import os
import random
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cf_infer import PolicyModel  # noqa: E402
from cf_search import Searcher  # noqa: E402

MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def make_engine(spec: str, min_depth: int, policy_cost: int) -> tuple[Searcher, PolicyModel | None]:
    if spec == "none":
        return Searcher(), None
    model = PolicyModel(spec)
    model.warm_up()
    engine = Searcher(policy_fn=model.priors, policy_min_depth=min_depth)
    engine.policy_node_cost = policy_cost
    return engine, model


def random_opening(rng: random.Random) -> chess.Board:
    while True:
        board = chess.Board()
        for _ in range(rng.randint(4, 8)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and len(list(board.legal_moves)) > 5:
            return board


def material_balance(board: chess.Board) -> int:
    bal = 0
    for pt, val in MATERIAL.items():
        bal += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return bal


Stats = dict[str, list[float]]


def play(
    white: Searcher,
    black: Searcher,
    start: chess.Board,
    nodes: int,
    max_plies: int,
    stats_white: Stats,
    stats_black: Stats,
) -> int:
    """Return +1 (white wins), 0 (draw), -1 (black wins)."""
    board = start.copy()
    for engine in (white, black):
        engine.tt.clear()
        engine.game_history.clear()
        engine.policy_cache.clear()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        engine = white if board.turn == chess.WHITE else black
        stats = stats_white if board.turn == chess.WHITE else stats_black
        engine.note_position(board)
        c0 = time.process_time()
        result = engine.search(board, budget_s=600.0, max_nodes=nodes)
        stats["cpu"].append(time.process_time() - c0)
        stats["nodes"].append(result.nodes)
        stats["depth"].append(result.depth)
        stats["policy_calls"].append(engine.policy_calls)
        if result.move is None:
            break
        board.push(result.move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return 0 if outcome.winner is None else (1 if outcome.winner == chess.WHITE else -1)
    bal = material_balance(board)
    return 0 if abs(bal) < 2 else (1 if bal > 0 else -1)


def elo(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=os.path.join(ROOT, "models", "chessformer.npz"))
    parser.add_argument("--b", default="none")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=4000)
    parser.add_argument("--min-depth", type=int, default=3, help="policy_min_depth for engines with a model")
    parser.add_argument("--policy-cost", type=int, default=-1, help="nodes charged per network call (-1: measure)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="append the summary to this file")
    args = parser.parse_args()

    policy_cost = args.policy_cost
    if policy_cost < 0:
        # measure: engine nodes/s without a model, and the latency of one network call
        probe = Searcher()
        board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8")
        c0 = time.process_time()
        res = probe.search(board, budget_s=600.0, max_nodes=6000)
        nps = res.nodes / max(1e-6, time.process_time() - c0)
        spec = args.a if args.a != "none" else args.b
        if spec == "none":
            policy_cost = 0
        else:
            model = PolicyModel(spec)
            model.warm_up()
            c0 = time.process_time()
            for _ in range(20):
                model.priors(board)
            latency = (time.process_time() - c0) / 20
            policy_cost = int(round(latency * nps))
            print(f"engine {nps:.0f} nodes/s (cpu), network {latency * 1000:.1f} ms -> policy cost {policy_cost} nodes")

    eng_a, _ = make_engine(args.a, args.min_depth, policy_cost)
    eng_b, _ = make_engine(args.b, args.min_depth, policy_cost)
    rng = random.Random(args.seed)
    wins = draws = losses = 0
    stats_a: Stats = {"cpu": [], "nodes": [], "depth": [], "policy_calls": []}
    stats_b: Stats = {"cpu": [], "nodes": [], "depth": [], "policy_calls": []}
    t0 = time.time()
    for pair in range(args.pairs):
        start = random_opening(rng)
        for a_is_white in (True, False):
            if a_is_white:
                r = play(eng_a, eng_b, start, args.nodes, args.max_plies, stats_a, stats_b)
            else:
                r = -play(eng_b, eng_a, start, args.nodes, args.max_plies, stats_b, stats_a)
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
        n = 2 * (pair + 1)
        score = (wins + 0.5 * draws) / n
        print(
            f"pair {pair + 1}/{args.pairs}: A +{wins} ={draws} -{losses} score {score:.1%} "
            f"elo {elo(score):+.0f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    n = 2 * args.pairs
    score = (wins + 0.5 * draws) / n
    # binomial-ish error bar on the score -> elo interval
    sd = math.sqrt(max(1e-9, score * (1 - score) / n))
    lo, hi = elo(score - 1.96 * sd), elo(score + 1.96 * sd)
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def per_move(name: str, s: Stats) -> str:
        return (
            f"{name} per move: cpu {mean(s['cpu']) * 1000:.0f} ms, nodes {mean(s['nodes']):.0f}, "
            f"depth {mean(s['depth']):.2f}, policy calls {mean(s['policy_calls']):.1f}\n"
        )

    summary = (
        f"A={args.a} B={args.b} nodes={args.nodes} min_depth={args.min_depth} policy_cost={policy_cost} "
        f"pairs={args.pairs} seed={args.seed}\n"
        f"A: +{wins} ={draws} -{losses} score {score:.1%} elo {elo(score):+.0f} [{lo:+.0f}, {hi:+.0f}]\n"
        + per_move("A", stats_a)
        + per_move("B", stats_b)
    )
    print(summary)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
