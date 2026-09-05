"""Paired self-play between two evaluation modes of the DeepChess agent at a fixed node
budget per move.

The machine that runs these sessions is shared with dozens of other arenas, so wall-clock
matches measure load rather than strength. Here every move gets the same number of search
nodes whichever side moves, every opening is played twice with colours swapped, and the
result is independent of how busy the machine is.

Modes are the values ``agent.EVAL_MODE`` accepts (``net``, ``hand``, ``blend``); a blend
weight can be appended, e.g. ``blend:0.75`` gives the network 75 % of the score.

Referee rules mirrored from ``harness/referee.py``: draws claimed automatically (threefold,
fifty moves, insufficient material), 300 plies goes to material adjudication.

    .venv/Scripts/python.exe agents/29_deepchess/tools/selfplay.py \
        --a net --b hand --nodes 3000 --games 40 --out agents/29_deepchess/results/net_vs_hand

The run is resumable: finished games are appended to ``games.jsonl`` in ``--out`` and the
summary is rewritten after each game.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
sys.path.insert(0, str(AGENT_DIR))

import agent  # noqa: E402

if agent._compile_thread is not None:
    agent._compile_thread.join()  # the compiled engine is needed before any game starts

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class NodeLimitedSearcher(agent.Searcher):
    """The agent's searcher, stopped by a node budget instead of the clock."""

    max_nodes = 10**9
    cpu_deadline = float("inf")

    def _check_time(self) -> None:
        if self.armed and (self.nodes >= self.max_nodes
                           or time.process_time() >= self.cpu_deadline):
            raise agent.OutOfTime()


class Player:
    """One side: ``[engine:]mode[:weight]`` with engine numba (default) or python.

    The budget is either ``nodes`` per move or, when ``cpu_ms`` is given, process CPU
    time per move: exact for the python engine (checked every node) and approximated for
    the compiled engine through a node cap derived from its measured CPU nodes/s.
    """

    def __init__(self, spec: str, nodes: int, cpu_ms: float | None) -> None:
        parts = spec.split(":")
        engine = "numba"
        if parts and parts[0] in ("numba", "python"):
            engine = parts.pop(0)
        mode = parts[0] if parts else "net"
        if mode not in ("net", "hand", "blend"):
            raise SystemExit(f"unknown mode {mode!r}; use net, hand or blend[:weight]")
        self.spec = spec
        self.engine = engine
        self.mode = mode
        self.weight = float(parts[1]) if len(parts) > 1 else 0.5
        self.nodes = nodes
        self.cpu_ms = cpu_ms
        self.searcher = NodeLimitedSearcher()
        self.numba: agent.NumbaSearcher | None = None
        if engine == "numba":
            if agent._MODEL is None:
                raise SystemExit("numba engine needs the model weights")
            self.numba = agent.NumbaSearcher(agent._MODEL)
        self.nps_cpu = 300_000.0
        self.depths: list[int] = []
        self.game_history: dict[int, int] = {}

    def new_game(self) -> None:
        self.searcher = NodeLimitedSearcher()
        self.game_history = {}
        if self.numba is not None:
            self.numba.new_game()

    def choose(self, board: chess.Board) -> chess.Move:
        agent.EVAL_MODE = self.mode
        agent.BLEND_NET_WEIGHT = self.weight
        if self.numba is not None:
            return self._choose_numba(board)
        return self._choose_python(board)

    def _choose_numba(self, board: chess.Board) -> chess.Move:
        ns = self.numba
        assert ns is not None
        budget = self.nodes
        if self.cpu_ms is not None:
            budget = max(2000, int(self.nps_cpu * self.cpu_ms / 1000.0))
        ns.prepare(board, max_nodes=budget)
        root = [(agent._move_code(board, m), m) for m in board.legal_moves]
        best = root[0][1]
        depth_reached = 0
        c0 = time.process_time()
        for depth in range(1, agent.dc_engine.MAX_PLY - 4):
            try:
                score, move, root = ns.search_root(depth, root)
            except agent.OutOfTime:
                break
            best = move
            depth_reached = depth
            if abs(score) > agent.MATE_BOUND and depth >= 4:
                break
            if ns.stats[agent.dc_search.ST_NODES] >= budget:
                break
        cpu = time.process_time() - c0
        nodes = int(ns.stats[agent.dc_search.ST_NODES])
        if cpu > 0.05 and nodes > 5000:
            self.nps_cpu = 0.7 * self.nps_cpu + 0.3 * nodes / cpu
        self.depths.append(depth_reached)
        return best

    def _choose_python(self, board: chess.Board) -> chess.Move:
        s = self.searcher
        if self.cpu_ms is not None:
            s.cpu_deadline = time.process_time() + self.cpu_ms / 1000.0
        key = board._transposition_key()
        self.game_history[key] = self.game_history.get(key, 0) + 1
        s.game_history = self.game_history
        s.nodes = s.qnodes = s.tt_hits = s.seldepth = 0
        s.path = {}
        for k in s.killers:
            k[0] = k[1] = None
        for hk in list(s.history):
            s.history[hk] //= 2
        moves = list(board.legal_moves)
        best = agent._quick_move(board, moves)
        ordered = moves
        s.max_nodes = self.nodes if self.cpu_ms is None else 10**9
        s.armed = True
        depth_reached = 0
        stack_len = len(board.move_stack)
        for depth in range(1, agent.MAX_PLY):
            try:
                score, move, ordered = s.search_root(board, depth, ordered, best)
            except agent.OutOfTime:
                while len(board.move_stack) > stack_len:
                    board.pop()
                break
            best = move
            depth_reached = depth
            if abs(score) > agent.MATE_BOUND and depth >= 4:
                break
            if s.nodes >= self.nodes:
                break
        self.depths.append(depth_reached)
        return best


def adjudicate(board: chess.Board) -> str:
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    return "1-0" if balance > 0 else ("0-1" if balance < 0 else "1/2-1/2")


def play(white: Player, black: Player, opening: list[str], ply_cap: int) -> tuple[str, str, int]:
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    white.new_game()
    black.new_game()
    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            assert outcome is not None
            return outcome.result(), outcome.termination.name.lower(), len(board.move_stack)
        if len(board.move_stack) >= ply_cap:
            return adjudicate(board), "adjudication", len(board.move_stack)
        player = white if board.turn == chess.WHITE else black
        board.push(player.choose(board))


def make_openings(count: int, seed: int, plies: int) -> list[list[str]]:
    """Random openings kept only when the handcrafted eval calls them roughly level."""
    rng = random.Random(seed)
    openings: list[list[str]] = []
    seen: set[str] = set()
    agent.EVAL_MODE = "hand"
    while len(openings) < count:
        board = chess.Board()
        line: list[str] = []
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            line.append(move.uci())
            board.push(move)
        if board.is_game_over() or len(line) < plies:
            continue
        if abs(agent.evaluate(board)) > 80:
            continue
        key = board.fen()
        if key in seen:
            continue
        seen.add(key)
        openings.append(line)
    return openings


def elo(score: float, n: int) -> tuple[float, float]:
    """Elo difference and its 95 % half-width from a score fraction over n games."""
    p = min(max(score, 1e-6), 1 - 1e-6)
    diff = -400 * math.log10(1 / p - 1)
    sd = math.sqrt(p * (1 - p) / max(n, 1))
    lo = min(max(p - 1.96 * sd, 1e-6), 1 - 1e-6)
    hi = min(max(p + 1.96 * sd, 1e-6), 1 - 1e-6)
    half = (-400 * math.log10(1 / hi - 1) + 400 * math.log10(1 / lo - 1)) / 2
    return diff, half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="net | hand | blend[:weight]")
    parser.add_argument("--b", required=True)
    parser.add_argument("--nodes", type=int, default=3000)
    parser.add_argument("--cpu-ms", type=float, default=None,
                        help="CPU time per move instead of a node budget")
    parser.add_argument("--games", type=int, default=40, help="total games (pairs x 2)")
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ply-cap", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "games.jsonl"
    done: list[dict] = []
    if log.exists():
        done = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    pa = Player(args.a, args.nodes, args.cpu_ms)
    pb = Player(args.b, args.nodes, args.cpu_ms)
    openings = make_openings((args.games + 1) // 2, args.seed, args.opening_plies)

    started = time.time()
    with log.open("a") as fh:
        for game in range(len(done), args.games):
            opening = openings[game // 2]
            a_white = game % 2 == 0
            white, black = (pa, pb) if a_white else (pb, pa)
            t0 = time.process_time()
            result, termination, plies = play(white, black, opening, args.ply_cap)
            if result == "1-0":
                a_score = 1.0 if a_white else 0.0
            elif result == "0-1":
                a_score = 0.0 if a_white else 1.0
            else:
                a_score = 0.5
            record = {
                "game": game, "opening": " ".join(opening), "a_white": a_white,
                "result": result, "termination": termination, "plies": plies,
                "a_score": a_score, "cpu_s": round(time.process_time() - t0, 1),
            }
            done.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            total = sum(r["a_score"] for r in done)
            print(f"game {game + 1}/{args.games}: {result} by {termination} in {plies} plies; "
                  f"{args.a} {total:.1f}/{len(done)}", flush=True)
            write_summary(args, done, pa, pb, time.time() - started)


def write_summary(args: argparse.Namespace, done: list[dict], pa: Player, pb: Player,
                  elapsed: float) -> None:
    n = len(done)
    wins = sum(1 for r in done if r["a_score"] == 1.0)
    draws = sum(1 for r in done if r["a_score"] == 0.5)
    losses = n - wins - draws
    score = (wins + draws / 2) / n if n else 0.0
    diff, half = elo(score, n) if n else (0.0, 0.0)
    terms: dict[str, int] = {}
    for r in done:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    summary = {
        "a": args.a, "b": args.b, "nodes": args.nodes, "cpu_ms": args.cpu_ms, "games": n,
        "a_wins": wins, "draws": draws, "a_losses": losses, "a_score": round(score, 4),
        "elo_a_minus_b": round(diff, 1), "elo_95_half_width": round(half, 1),
        "terminations": terms,
        "mean_depth_a": round(sum(pa.depths) / len(pa.depths), 2) if pa.depths else None,
        "mean_depth_b": round(sum(pb.depths) / len(pb.depths), 2) if pb.depths else None,
        "elapsed_s": round(elapsed, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
