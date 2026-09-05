"""Generate training positions by engine self-play, labelled by the engine's own search.

Teacher: cf_search.Searcher (this project's alpha-beta) at a fixed depth. Every root position
is labelled with the searched best move and score; in addition every transposition-table
*exact* node of depth >= --harvest-depth found during the search is harvested with its best
move and score (these are principal-variation positions, so they are "real" chess positions).

Diversity: random opening plies, and an epsilon chance of playing a random move instead of the
engine move (the label is always the engine's best move, never the random move).

Output: one .npz shard per worker chunk in --out, with arrays
    x      uint8  [N, 67]   compact position (see cf_encode.compact)
    move   int16  [N]       policy index of the teacher's best move
    score  int16  [N]       teacher score, centipawns, side to move, clipped to +-3000
    depth  int8   [N]       search depth of the label
    result int8   [N]       game result from the side to move's view (+1 / 0 / -1)
Provenance: seeds, depth and epsilon are stored in the shard as well.

Usage:
    python training/gen_data.py --out training/data --workers 6 --positions 300000 --depth 4
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cf_encode import COMPACT_SIZE, compact, move_index  # noqa: E402
from cf_eval import MATE_BOUND  # noqa: E402
from cf_search import Searcher  # noqa: E402


def _clip_score(score: int) -> int:
    if score > MATE_BOUND:
        return 3000
    if score < -MATE_BOUND:
        return -3000
    return max(-3000, min(3000, score))


def play_game(
    searcher: Searcher,
    rng: random.Random,
    depth: int,
    budget_s: float,
    epsilon: float,
    harvest_depth: int,
    max_plies: int,
) -> tuple[list[tuple[np.ndarray, int, int, int, bool]], int]:
    """Return (positions, result) where result is +1 white win, -1 black win, 0 draw.

    Each position: (compact, move_index, score, depth, white_to_move).
    """
    board = chess.Board()
    for _ in range(rng.randint(2, 10)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    if board.is_game_over():
        board = chess.Board()
    searcher.tt.clear()
    searcher.game_history.clear()
    searcher.harvest_min_depth = harvest_depth
    positions: list[tuple[np.ndarray, int, int, int, bool]] = []
    seen: set[str] = set()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        searcher.note_position(board)
        searcher.harvest = []
        result = searcher.search(board, budget_s, max_depth=depth)
        if result.move is None:
            break
        key = board.fen()
        if key not in seen:
            seen.add(key)
            positions.append(
                (
                    compact(board),
                    move_index(result.move, not board.turn),
                    _clip_score(result.score),
                    result.depth,
                    board.turn,
                )
            )
        for fen, best_uci, score, hdepth in searcher.harvest:
            if fen in seen:
                continue
            seen.add(fen)
            b2 = chess.Board(fen)
            mv = chess.Move.from_uci(best_uci)
            positions.append(
                (compact(b2), move_index(mv, not b2.turn), _clip_score(score), hdepth, b2.turn)
            )
        searcher.harvest = None
        if rng.random() < epsilon:
            move = rng.choice(list(board.legal_moves))
        else:
            move = result.move
        board.push(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        # adjudicate long games on material like the harness does
        if plies >= max_plies:
            bal = 0
            for pt, val in ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)):
                bal += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
            game_result = 1 if bal > 0 else (-1 if bal < 0 else 0)
        else:
            game_result = 0
    else:
        game_result = 1 if outcome.winner == chess.WHITE else -1
    return positions, game_result


def worker(args: tuple[int, int, str, int, float, float, int, int]) -> str:
    seed, n_positions, out_path, depth, budget_s, epsilon, harvest_depth, max_plies = args
    rng = random.Random(seed)
    searcher = Searcher()
    xs: list[np.ndarray] = []
    moves: list[int] = []
    scores: list[int] = []
    depths: list[int] = []
    results: list[int] = []
    games = 0
    t0 = time.time()
    while len(xs) < n_positions:
        positions, game_result = play_game(
            searcher, rng, depth, budget_s, epsilon, harvest_depth, max_plies
        )
        games += 1
        for cx, mi, sc, dp, white_to_move in positions:
            xs.append(cx)
            moves.append(mi)
            scores.append(sc)
            depths.append(dp)
            results.append(game_result if white_to_move else -game_result)
        if games % 5 == 0:
            rate = len(xs) / max(1e-9, time.time() - t0)
            print(f"[worker {seed}] games {games} positions {len(xs)} ({rate:.0f}/s)", flush=True)
    np.savez_compressed(
        out_path,
        x=np.stack(xs).astype(np.uint8),
        move=np.array(moves, dtype=np.int16),
        score=np.array(scores, dtype=np.int16),
        depth=np.array(depths, dtype=np.int8),
        result=np.array(results, dtype=np.int8),
        seed=np.array([seed]),
        params=np.array([depth, budget_s, epsilon, harvest_depth, max_plies]),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "data"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--positions", type=int, default=300_000, help="total positions")
    parser.add_argument("--chunk", type=int, default=25_000, help="positions per shard")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--budget", type=float, default=3.0, help="seconds cap per search")
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--harvest-depth", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_chunks = max(1, args.positions // args.chunk)
    jobs = []
    for i in range(n_chunks):
        seed = args.seed + i
        path = os.path.join(args.out, f"shard_d{args.depth}_s{seed}.npz")
        if os.path.exists(path):
            continue
        jobs.append(
            (seed, args.chunk, path, args.depth, args.budget, args.epsilon, args.harvest_depth, args.max_plies)
        )
    print(f"{len(jobs)} shards to generate with {args.workers} workers", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        for job in jobs:
            print(f"wrote {worker(job)} ({time.time() - t0:.0f}s)", flush=True)
        return
    with mp.Pool(args.workers) as pool:
        for path in pool.imap_unordered(worker, jobs):
            print(f"wrote {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
