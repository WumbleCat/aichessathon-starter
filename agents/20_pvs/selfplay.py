"""Paired self-play between two parameter sets at a fixed node budget per move.

Machine load does not change the result because moves are limited by nodes, not time.
Every opening is played twice with colours swapped. Draw rules mirror the referee:
threefold repetition, fifty moves, insufficient material, and material adjudication after
300 plies. The run is resumable: finished games are appended to ``<out>.jsonl`` and the
summary is recomputed from that file.

    python agents/20_pvs/selfplay.py --b P_NULL=0 --nodes 4000 --games 100
    python agents/20_pvs/selfplay.py --a P_RFP_MARGIN=70 --b P_RFP_MARGIN=100 --games 200

``--a`` and ``--b`` take comma-separated ``P_NAME=value`` overrides of ``default_params``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import numpy as np
import pvs_search as ps
from pvs_board import ST_HASH, Position, move_to_uci

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PLY_CAP = 300


def parse_params(spec: str) -> np.ndarray:
    params = ps.default_params()
    for item in filter(None, spec.split(",")):
        name, value = item.split("=")
        params[getattr(ps, name.strip())] = int(value)
    return params


def material(board: chess.Board, colour: chess.Color) -> int:
    return sum(len(board.pieces(pt, colour)) * v for pt, v in PIECE_VALUES.items())


def random_opening(rng: random.Random, plies: int) -> str:
    """A random legal line of ``plies`` half-moves that leaves both sides some material."""
    while True:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or abs(material(board, True) - material(board, False)) > 2:
            continue
        return board.fen()


def play_game(
    fen: str, white: ps.Searcher, black: ps.Searcher, nodes: int
) -> tuple[str, str, int]:
    """Return (result 'white'|'black'|'draw', termination, plies)."""
    board = chess.Board(fen)
    pos = Position(fen)
    white.clear()
    black.clear()
    keys: list[int] = []
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return "draw", outcome.termination.name.lower(), board.ply()
            return ("white" if outcome.winner else "black"), "checkmate", board.ply()
        if board.ply() - chess.Board(fen).ply() >= PLY_CAP:
            diff = material(board, chess.WHITE) - material(board, chess.BLACK)
            if diff > 0:
                return "white", "adjudication", board.ply()
            if diff < 0:
                return "black", "adjudication", board.ply()
            return "draw", "adjudication", board.ply()
        searcher = white if board.turn == chess.WHITE else black
        pos.set_fen(board.fen())
        keys.append(int(pos.st[ST_HASH]))
        move, _score, _depth, _info = searcher.search(
            pos, node_limit=nodes, max_depth=64, history_keys=keys
        )
        uci = move_to_uci(move)
        assert chess.Move.from_uci(uci) in board.legal_moves, (uci, board.fen())
        board.push_uci(uci)


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def summarise(games: list[dict]) -> str:
    n = len(games)
    if n == 0:
        return "no games"
    w = sum(1 for g in games if g["a"] == "win")
    d = sum(1 for g in games if g["a"] == "draw")
    losses = n - w - d
    score = (w + d / 2) / n
    # standard error of the score from the per-game outcomes
    values = [1.0 if g["a"] == "win" else 0.5 if g["a"] == "draw" else 0.0 for g in games]
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.5
    se = sd / math.sqrt(n)
    lo, hi = elo(max(score - 1.96 * se, 1e-6)), elo(min(score + 1.96 * se, 1 - 1e-6))
    terms: dict[str, int] = {}
    for g in games:
        terms[g["termination"]] = terms.get(g["termination"], 0) + 1
    plies = statistics_mean([g["plies"] for g in games])
    return (
        f"A vs B: +{w} ={d} -{losses} over {n} games, score {score:.1%}, "
        f"elo {elo(score):+.0f} (95% {lo:+.0f}..{hi:+.0f}), avg plies {plies:.0f}, "
        + ", ".join(f"{k} {v}" for k, v in sorted(terms.items()))
    )


def statistics_mean(values: list[int]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="", help="P_NAME=value,... for side A")
    parser.add_argument("--b", default="", help="P_NAME=value,... for side B")
    parser.add_argument("--nodes", type=int, default=4000)
    parser.add_argument("--games", type=int, default=100, help="total games (2 per opening)")
    parser.add_argument("--plies", type=int, default=6, help="random opening length")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="", help="jsonl file for resumable results")
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "selfplay_last.jsonl"
    )
    games: list[dict] = []
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            games = [json.loads(line) for line in fh if line.strip()]
        print(f"resuming from {len(games)} games in {out}")
    rng = random.Random(args.seed)
    openings = [random_opening(rng, args.plies) for _ in range((args.games + 1) // 2)]

    a = ps.Searcher(parse_params(args.a))
    b = ps.Searcher(parse_params(args.b))
    started = time.perf_counter()
    cpu0 = time.process_time()
    for index in range(len(games), args.games):
        fen = openings[index // 2]
        a_white = index % 2 == 0
        white, black = (a, b) if a_white else (b, a)
        result, termination, plies = play_game(fen, white, black, args.nodes)
        if result == "draw":
            a_result = "draw"
        else:
            a_result = "win" if (result == "white") == a_white else "loss"
        record = {
            "index": index, "fen": fen, "a_white": a_white, "result": result,
            "termination": termination, "plies": plies, "a": a_result,
        }
        games.append(record)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"game {index + 1}/{args.games}: A {'white' if a_white else 'black'} -> {a_result} "
            f"({termination}, {plies} plies)  |  {summarise(games)}"
        )
    print(
        f"\n{summarise(games)}\nwall {time.perf_counter() - started:.0f}s "
        f"cpu {time.process_time() - cpu0:.0f}s nodes/move {args.nodes} a='{args.a}' b='{args.b}'"
    )


if __name__ == "__main__":
    main()
