"""Generate policy training data with this project's own search as the teacher.

Each worker plays noisy self-play games. At every position the teacher runs an iterative
deepening search that scores EVERY legal root move with a full window (``want_root_scores``),
so the label is a full distribution over legal moves, not just the best one. The next move is
sampled from softmax(scores / T) with some uniform noise so the games stay diverse.

Shard format (npz):
    codes        uint8  (N, 64)   piece codes, true frame (see pn_encoding.board_to_codes)
    meta         int32  (N,)      turn / castling / ep-file bits
    n_moves      int16  (N,)      number of labelled moves for the position
    best         int16  (N,)      action index of the teacher's best move (STM frame)
    value        int16  (N,)      teacher score of the best move, STM point of view, clipped
    result       int8   (N,)      game result from STM point of view (+1 / 0 / -1)
    ply          int16  (N,)
    depth        int8   (N,)      teacher depth reached
    label_pos    int32  (M,)      index into the position arrays
    label_action int16  (M,)      action index (STM frame)
    label_score  int16  (M,)      teacher score for that move, STM point of view, clipped

Usage:
    python training/gen_data.py --workers 8 --positions 200000 --out training/data
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402
from pn_encoding import board_to_codes, move_to_index  # noqa: E402
from pn_search import Searcher  # noqa: E402

SCORE_CLIP = 3000
ADJUDICATE_CP = 1000
ADJUDICATE_PLIES = 4
MAX_PLIES = 240


def _sample_move(
    scores: dict[chess.Move, int], temperature: float, rng: random.Random
) -> chess.Move:
    moves = list(scores)
    best = max(scores.values())
    weights = [math.exp(max(-30.0, (scores[m] - best) / temperature)) for m in moves]
    return rng.choices(moves, weights=weights, k=1)[0]


def _opening(rng: random.Random) -> chess.Board:
    """Random opening: the start position or a few random plies from it."""
    board = chess.Board()
    plies = rng.choice([0, 0, 2, 4, 6, 8, 10, 12])
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        # prefer sensible-looking random moves: no early queen sorties, some captures
        board.push(rng.choice(moves))
        if board.is_game_over():
            return chess.Board()
    return board


class ShardWriter:
    def __init__(self, out_dir: str, worker: int, shard_size: int, seed: int = 0) -> None:
        self.out_dir = out_dir
        self.worker = worker
        self.seed = seed
        self.shard_size = shard_size
        self.shard_index = 0
        self.reset()

    def reset(self) -> None:
        self.codes: list[np.ndarray] = []
        self.meta: list[int] = []
        self.n_moves: list[int] = []
        self.best: list[int] = []
        self.value: list[int] = []
        self.result: list[int] = []
        self.ply: list[int] = []
        self.depth: list[int] = []
        self.label_pos: list[int] = []
        self.label_action: list[int] = []
        self.label_score: list[int] = []

    def __len__(self) -> int:
        return len(self.meta)

    def add_game(self, records: list[dict], result_white: float) -> None:
        for rec in records:
            pos = len(self.meta)
            stm_result = result_white if rec["turn"] else -result_white
            self.codes.append(rec["codes"])
            self.meta.append(rec["meta"])
            self.n_moves.append(len(rec["labels"]))
            self.best.append(rec["best"])
            self.value.append(rec["value"])
            self.result.append(round(stm_result))
            self.ply.append(rec["ply"])
            self.depth.append(rec["depth"])
            for action, score in rec["labels"]:
                self.label_pos.append(pos)
                self.label_action.append(action)
                self.label_score.append(score)
        if len(self.meta) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.meta:
            return
        path = os.path.join(
            self.out_dir, f"shard_s{self.seed}_w{self.worker:02d}_{self.shard_index:04d}.npz"
        )
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            codes=np.stack(self.codes).astype(np.uint8),
            meta=np.array(self.meta, dtype=np.int32),
            n_moves=np.array(self.n_moves, dtype=np.int16),
            best=np.array(self.best, dtype=np.int16),
            value=np.array(self.value, dtype=np.int16),
            result=np.array(self.result, dtype=np.int8),
            ply=np.array(self.ply, dtype=np.int16),
            depth=np.array(self.depth, dtype=np.int8),
            label_pos=np.array(self.label_pos, dtype=np.int32),
            label_action=np.array(self.label_action, dtype=np.int16),
            label_score=np.array(self.label_score, dtype=np.int16),
        )
        os.replace(tmp, path)
        self.shard_index += 1
        self.reset()


def worker_main(worker: int, args: argparse.Namespace, counter, stop_flag) -> None:
    rng = random.Random(args.seed * 1000 + worker)
    searcher = Searcher()
    writer = ShardWriter(args.out, worker, args.shard_size, args.seed)
    games = 0
    while not stop_flag.value:
        board = _opening(rng)
        searcher.new_game()
        records: list[dict] = []
        extreme = 0
        result_white = 0.0
        ply = 0
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.winner is None:
                    result_white = 0.0
                else:
                    result_white = 1.0 if outcome.winner == chess.WHITE else -1.0
                break
            if ply >= MAX_PLIES:
                result_white = 0.0
                break
            searcher.remember_position(board)
            res = searcher.search(
                board,
                args.depth,
                time_budget=args.time,
                want_root_scores=True,
                node_limit=args.nodes,
            )
            scores = res.root_scores
            if not scores or res.move is None:
                # depth 1 always completes, so this only happens with no legal moves
                break
            flip = board.turn == chess.BLACK
            best_move = max(scores, key=scores.get)
            best_score = scores[best_move]
            codes, meta = board_to_codes(board)
            labels = [
                (move_to_index(m, flip), int(max(-SCORE_CLIP, min(SCORE_CLIP, s))))
                for m, s in scores.items()
            ]
            records.append(
                {
                    "codes": codes,
                    "meta": meta,
                    "turn": board.turn,
                    "best": move_to_index(best_move, flip),
                    "value": int(max(-SCORE_CLIP, min(SCORE_CLIP, best_score))),
                    "ply": ply,
                    "depth": res.depth,
                    "labels": labels,
                }
            )
            # adjudicate hopeless games
            if abs(best_score) >= ADJUDICATE_CP:
                extreme += 1
                if extreme >= ADJUDICATE_PLIES:
                    winner_is_stm = best_score > 0
                    result_white = 1.0 if (winner_is_stm == (board.turn == chess.WHITE)) else -1.0
                    break
            else:
                extreme = 0
            # choose the move to play
            if rng.random() < args.random_move:
                move = rng.choice(list(scores))
            else:
                temperature = args.temp_opening if ply < 16 else args.temp
                move = _sample_move(scores, temperature, rng)
            board.push(move)
            ply += 1
        writer.add_game(records, result_white)
        games += 1
        with counter.get_lock():
            counter.value += len(records)
            total = counter.value
        if total >= args.positions:
            stop_flag.value = 1
    writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--positions", type=int, default=100_000)
    parser.add_argument("--out", default=os.path.join(HERE, "data"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--depth", type=int, default=4, help="teacher max depth")
    parser.add_argument("--nodes", type=int, default=3500, help="teacher node limit per position")
    parser.add_argument(
        "--time", type=float, default=3.0, help="teacher wall time cap per position"
    )
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--temp", type=float, default=40.0)
    parser.add_argument("--temp-opening", type=float, default=80.0)
    parser.add_argument("--random-move", type=float, default=0.02)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    counter = mp.Value("i", 0)
    stop_flag = mp.Value("i", 0)
    procs = [
        mp.Process(target=worker_main, args=(w, args, counter, stop_flag), daemon=True)
        for w in range(args.workers)
    ]
    for p in procs:
        p.start()
    started = time.time()
    last = 0
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(30)
            total = counter.value
            elapsed = time.time() - started
            rate = (total - last) / 30.0
            last = total
            print(f"[{elapsed / 60:6.1f} min] positions {total:8d}  ({rate:5.1f}/s)", flush=True)
    except KeyboardInterrupt:
        stop_flag.value = 1
    for p in procs:
        p.join()
    print("done, positions:", counter.value)


if __name__ == "__main__":
    main()
