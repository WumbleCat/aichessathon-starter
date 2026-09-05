"""Generate NNUE training positions with an OFFLINE Stockfish teacher.

Each worker owns one Stockfish process and plays games: a random opening of 4..12 plies, then
Stockfish moves at a fixed node budget with occasional random moves for diversity.  The
teacher's own search score at each ply is the label (side-to-move relative centipawns).

Filtered out: positions in check, positions whose best move is a capture/promotion (not
quiet, so a leaf evaluator should not be trained on them), the random opening plies, and
mate scores (clipped to +-MAX_CP).

Output: one text line per position ``<fen>,<stm_cp>,<result_for_stm>,<ply>`` appended to
``data/positions_<worker>.txt``.  Provenance: this file, the seed and the teacher version.

Usage:
    python training/datagen.py --workers 6 --positions 1000000 --nodes 8000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Process

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER = os.path.join(HERE, "teacher", "stockfish", "stockfish-windows-x86-64-avx2.exe")
DATA_DIR = os.path.join(HERE, "data")
MAX_CP = 3000
MAX_PLIES = 400


def worker(index: int, target: int, nodes: int, seed: int, out_path: str) -> None:
    rng = random.Random(seed * 1000 + index)
    engine = chess.engine.SimpleEngine.popen_uci(TEACHER)
    engine.configure({"Threads": 1, "Hash": 32})
    written = 0
    games = 0
    t0 = time.time()
    limit = chess.engine.Limit(nodes=nodes)
    with open(out_path, "a", encoding="utf-8") as out:
        while written < target:
            board = chess.Board()
            opening_plies = rng.randint(4, 12)
            for _ in range(opening_plies):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if board.is_game_over():
                continue
            records: list[tuple[str, int, int]] = []  # fen, stm cp, side to move (True white)
            random_move_prob = rng.choice([0.0, 0.02, 0.05, 0.1])
            adjudicated: int | None = None  # +1 white wins, -1 black, 0 draw
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
                if rng.random() < random_move_prob:
                    board.push(rng.choice(list(board.legal_moves)))
                    continue
                result = engine.play(board, limit, info=chess.engine.INFO_SCORE)
                move = result.move
                if move is None:
                    break
                score = result.info.get("score")
                if score is not None:
                    pov = score.pov(board.turn)
                    if pov.is_mate():
                        cp = MAX_CP if pov.mate() > 0 else -MAX_CP
                    else:
                        cp = max(-MAX_CP, min(MAX_CP, pov.score()))
                    quiet = not board.is_capture(move) and move.promotion is None and not board.is_check()
                    if quiet:
                        records.append((board.fen(), cp, board.turn))
                    # adjudicate hopeless games to save teacher time
                    white_cp = cp if board.turn else -cp
                    if abs(white_cp) >= 2500 and len(board.move_stack) > 40:
                        adjudicated = 1 if white_cp > 0 else -1
                        board.push(move)
                        break
                board.push(move)
            games += 1
            if adjudicated is not None:
                white_result = adjudicated
            else:
                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    white_result = 0
                else:
                    white_result = 1 if outcome.winner else -1
            for fen, cp, turn in records:
                res = white_result if turn else -white_result
                out.write(f"{fen},{cp},{res},{len(board.move_stack)}\n")
                written += 1
            out.flush()
            if games % 10 == 0:
                rate = written / max(1e-9, time.time() - t0)
                print(f"[worker {index}] games {games} positions {written} ({rate:.0f}/s)", file=sys.stderr, flush=True)
    engine.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--positions", type=int, default=1_000_000, help="total positions")
    parser.add_argument("--nodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--tag", default="sf")
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    per_worker = args.positions // args.workers
    procs = []
    for i in range(args.workers):
        out_path = os.path.join(DATA_DIR, f"positions_{args.tag}_{i}.txt")
        p = Process(target=worker, args=(i, per_worker, args.nodes, args.seed, out_path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
