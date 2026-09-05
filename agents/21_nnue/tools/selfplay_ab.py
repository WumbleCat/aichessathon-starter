"""Node-limited paired self-play: NNUE evaluation vs the PSQT fallback (or vs another net).

Everything runs in one process, so numba compiles once and the result does not depend on how
loaded the machine is (each side gets the same node budget per move, not wall time).  Each
opening is played twice with colours swapped.

    python tools/selfplay_ab.py --games 40 --nodes 20000 --out results/ab_nnue_vs_psqt.txt
    python tools/selfplay_ab.py --games 40 --movetime 0.2   # equal wall time per move
    python tools/selfplay_ab.py --games 40 --weights-b models/other.safetensors  # net A vs net B
"""

from __future__ import annotations

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

import cboard as cb  # noqa: E402
import csearch  # noqa: E402
import nnue  # noqa: E402

PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def material_white(board: chess.Board) -> int:
    return sum(
        cp * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        for pt, cp in PIECE_CP.items()
    )


def random_opening(rng: random.Random, plies: int) -> chess.Board:
    while True:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and abs(material_white(board)) <= 300:
            return board


def play(
    white: csearch.Searcher,
    black: csearch.Searcher,
    start: chess.Board,
    nodes: int,
    movetime: float,
    ply_cap: int,
) -> tuple[float, str, dict[str, float]]:
    """Returns (white score, termination, stats) with stats nodes/seconds per side."""
    board = start.copy()
    keys: list[int] = []
    stats = {
        "w_nodes": 0.0,
        "w_time": 0.0,
        "b_nodes": 0.0,
        "b_time": 0.0,
        "w_cpu": 0.0,
        "b_cpu": 0.0,
    }
    while True:
        if board.is_checkmate():
            return (0.0 if board.turn else 1.0), "mate", stats
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.5, "draw", stats
        if len(board.move_stack) >= ply_cap:
            m = material_white(board)
            return (1.0 if m > 150 else 0.0 if m < -150 else 0.5), "adjudicated", stats
        side = white if board.turn else black
        side.set_position(board, keys)
        t0 = time.perf_counter()
        c0 = time.process_time()
        if movetime > 0:
            mv, _score, _depth, _pv, st = side.search(time_budget=movetime)
        else:
            mv, _score, _depth, _pv, st = side.search(node_limit=nodes)
        dt = time.perf_counter() - t0
        dc = time.process_time() - c0
        tag = "w" if board.turn else "b"
        stats[f"{tag}_nodes"] += st["nodes"]
        stats[f"{tag}_time"] += dt
        stats[f"{tag}_cpu"] += dc
        move = chess.Move.from_uci(cb.move_to_uci(mv)) if mv else None
        if move is None or move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        keys.append(int(side.P[cb.HASH]))
        board.push(move)


def elo(score: float, n: int) -> str:
    if n == 0 or score <= 0 or score >= n:
        return "n/a"
    p = score / n
    return f"{-400 * math.log10(1 / p - 1):+.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40, help="total games (pairs of two)")
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument(
        "--movetime", type=float, default=0.0, help="seconds per move; overrides --nodes"
    )
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--ply-cap", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", default=nnue.default_weights_path())
    parser.add_argument("--weights-b", default="", help="opponent net; default is the PSQT eval")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "ab_nnue_vs_psqt.txt"))
    args = parser.parse_args()

    net = nnue.load_net(args.weights)
    a = csearch.Searcher(net, use_nnue=True)  # NNUE
    if args.weights_b:
        b = csearch.Searcher(nnue.load_net(args.weights_b), use_nnue=True)
    else:
        b = csearch.Searcher(None, use_nnue=False)  # PSQT
    b_name = os.path.basename(args.weights_b) if args.weights_b else "psqt"
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t_start = time.perf_counter()
    c_start = time.process_time()
    score = 0.0
    wins = draws = losses = 0
    tot = {
        "nnue_nodes": 0.0,
        "nnue_time": 0.0,
        "nnue_cpu": 0.0,
        "psqt_nodes": 0.0,
        "psqt_time": 0.0,
        "psqt_cpu": 0.0,
    }
    with open(args.out, "a", encoding="utf-8") as out:
        out.write(
            f"\n# {time.strftime('%Y-%m-%d %H:%M')} "
            f"nnue={os.path.basename(args.weights)} vs {b_name} "
            f"nodes={args.nodes} movetime={args.movetime} games={args.games} seed={args.seed}\n"
        )
        for pair in range(args.games // 2):
            start = random_opening(rng, args.opening_plies)
            for nnue_white in (True, False):
                a.clear()
                b.clear()
                w, bl = (a, b) if nnue_white else (b, a)
                ws, term, st = play(w, bl, start, args.nodes, args.movetime, args.ply_cap)
                s = ws if nnue_white else 1.0 - ws
                score += s
                if s == 1.0:
                    wins += 1
                elif s == 0.0:
                    losses += 1
                else:
                    draws += 1
                nt, pt = ("w", "b") if nnue_white else ("b", "w")
                tot["nnue_nodes"] += st[f"{nt}_nodes"]
                tot["nnue_time"] += st[f"{nt}_time"]
                tot["psqt_nodes"] += st[f"{pt}_nodes"]
                tot["psqt_time"] += st[f"{pt}_time"]
                tot["nnue_cpu"] += st[f"{nt}_cpu"]
                tot["psqt_cpu"] += st[f"{pt}_cpu"]
                n = wins + draws + losses
                line = (
                    f"game {n:3d} pair {pair + 1} nnue={'white' if nnue_white else 'black'} "
                    f"result={s:.1f} by {term}  running +{wins} ={draws} -{losses} "
                    f"({score / n:.1%}, elo {elo(score, n)})"
                )
                print(line, flush=True)
                out.write(line + "\n")
                out.flush()
        n = wins + draws + losses
        nps_n = tot["nnue_nodes"] / max(1e-9, tot["nnue_cpu"])
        nps_p = tot["psqt_nodes"] / max(1e-9, tot["psqt_cpu"])
        wall_n = tot["nnue_nodes"] / max(1e-9, tot["nnue_time"])
        wall_p = tot["psqt_nodes"] / max(1e-9, tot["psqt_time"])
        summary = (
            f"TOTAL {os.path.basename(args.weights)} vs {b_name}: +{wins} ={draws} -{losses} "
            f"score {score / n:.1%} elo {elo(score, n)} | "
            f"nps by cpu A {nps_n:,.0f} B {nps_p:,.0f} (by wall {wall_n:,.0f} / {wall_p:,.0f}) | "
            f"wall {time.perf_counter() - t_start:.0f}s "
            f"cpu {time.process_time() - c_start:.0f}s"
        )
        print(summary, flush=True)
        out.write(summary + "\n")


if __name__ == "__main__":
    main()
