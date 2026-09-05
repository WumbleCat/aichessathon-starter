"""Play the compiled engine against another agent directory's ``get_move`` in one process.

Both sides get the same wall time per move (``--movetime``), which on a loaded machine is
noisy but symmetric; the compile happens once, so unlike harness games the engine plays from
move one.  Colours alternate on the same random openings.

    python tools/vs_bot.py --bot ../../my-agents/10_principal_variation_search --games 10
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from types import ModuleType

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from selfplay_ab import elo, material_white, random_opening  # noqa: E402

import cboard as cb  # noqa: E402
import csearch  # noqa: E402
import nnue  # noqa: E402


def load_bot(directory: str) -> ModuleType:
    path = os.path.join(os.path.abspath(directory), "agent.py")
    spec = importlib.util.spec_from_file_location("opponent_agent", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", required=True, help="directory with an agent.py")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--movetime", type=float, default=0.5, help="seconds per move, both sides")
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--ply-cap", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", default=nnue.default_weights_path())
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "vs_bots.txt"))
    args = parser.parse_args()

    bot = load_bot(args.bot)
    bot_name = os.path.basename(os.path.normpath(args.bot))
    engine = csearch.Searcher(nnue.load_net(args.weights), use_nnue=True)
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wins = draws = losses = 0
    score = 0.0
    with open(args.out, "a", encoding="utf-8") as out:
        out.write(
            f"\n# {time.strftime('%Y-%m-%d %H:%M')} 21_nnue ({os.path.basename(args.weights)}) vs "
            f"{bot_name} movetime={args.movetime} games={args.games} seed={args.seed}\n"
        )
        for g in range(args.games):
            start = random_opening(rng, args.opening_plies)
            engine_white = g % 2 == 0
            board = start.copy()
            engine.clear()
            keys: list[int] = []
            term = "ply-cap"
            while True:
                if board.is_checkmate():
                    ws, term = (0.0 if board.turn else 1.0), "mate"
                    break
                if (
                    board.is_stalemate()
                    or board.is_insufficient_material()
                    or board.can_claim_draw()
                ):
                    ws, term = 0.5, "draw"
                    break
                if len(board.move_stack) >= args.ply_cap:
                    m = material_white(board)
                    ws = 1.0 if m > 150 else 0.0 if m < -150 else 0.5
                    term = "adjudicated"
                    break
                if board.turn == engine_white:
                    engine.set_position(board, keys)
                    mv, _s, _d, _pv, _st = engine.search(time_budget=args.movetime)
                    move = chess.Move.from_uci(cb.move_to_uci(mv)) if mv else None
                    keys.append(int(engine.P[cb.HASH]))
                else:
                    uci = bot.get_move(board.fen(), int(args.movetime * 1000 * 30))
                    move = chess.Move.from_uci(uci)
                if move is None or move not in board.legal_moves:
                    move = next(iter(board.legal_moves))
                board.push(move)
            s = ws if engine_white else 1.0 - ws
            score += s
            wins += s == 1.0
            draws += s == 0.5
            losses += s == 0.0
            n = g + 1
            line = (
                f"game {n:2d} engine={'white' if engine_white else 'black'} "
                f"result={s:.1f} by {term} "
                f"({len(board.move_stack)} plies)  running +{wins} ={draws} -{losses} "
                f"({score / n:.1%}, elo {elo(score, n)})"
            )
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
        summary = (
            f"TOTAL 21_nnue vs {bot_name}: +{wins} ={draws} -{losses} "
            f"score {score / n:.1%} elo {elo(score, n)}"
        )
        print(summary, flush=True)
        out.write(summary + "\n")


if __name__ == "__main__":
    main()
