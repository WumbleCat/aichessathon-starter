"""Generate DeepChess training positions labelled by Stockfish (offline, training only).

Each worker plays games from the standard start position: a few random plies for
diversity, then moves chosen by a shallow Stockfish search with occasional random moves.
Every position after the random opening is labelled with the side-to-move score of a
fixed-depth Stockfish search. Nothing produced here ships except the network trained on
it; the engine binary lives in ``teacher/`` which is git-ignored and never packaged.

Output: ``data/chunk_<worker>_<n>.npz`` with arrays
  boards   uint8 (N, 64)   0 empty, 1..6 white PNBRQK, 7..12 black PNBRQK, a1 = 0
  meta     uint8 (N, 5)    turn (1 white), castling bits (KQkq = 1,2,4,8), ep flag,
                           in_check, best-move-is-capture
  scores   int16 (N,)      teacher cp from the side to move's point of view, clipped
  results  int8  (N,)      game result from White's view: 1, 0, -1
  games    int32 (N,)      game id (unique per worker)
  plies    int16 (N,)      ply index within the game
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEACHER = ROOT / "teacher" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
DATA = ROOT / "data"

SCORE_CLIP = 3000
MATE_SCORE = 3000
PIECE_CODE = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5,
              chess.KING: 6}


def encode_board(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    squares = np.zeros(64, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        squares[sq] = PIECE_CODE[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    meta = np.array([1 if board.turn else 0, castling,
                     1 if board.ep_square is not None else 0, 0, 0], dtype=np.uint8)
    return squares, meta


class Teacher:
    """Minimal UCI client. python-chess's engine wrapper costs ~30 ms per call; this ~1 ms."""

    def __init__(self, path: Path, hash_mb: int = 16) -> None:
        self.proc = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix: str) -> list[str]:
        assert self.proc.stdout is not None
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("teacher died")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def analyse(self, fen: str, depth: int) -> tuple[int, str | None]:
        """(score cp from side to move, best move uci or None)."""
        self._send("position fen " + fen)
        self._send(f"go depth {depth}")
        lines = self._read_until("bestmove")
        cp = 0
        for line in lines:
            if not line.startswith("info") or " score " not in line or " multipv 2" in line:
                continue
            parts = line.split()
            i = parts.index("score")
            kind, value = parts[i + 1], int(parts[i + 2])
            if kind == "cp":
                cp = max(-SCORE_CLIP, min(SCORE_CLIP, value))
            else:  # mate
                cp = MATE_SCORE if value > 0 else -MATE_SCORE
        best = lines[-1].split()[1]
        if best in ("(none)", "0000"):
            return cp, None
        return cp, best

    def quit(self) -> None:
        self._send("quit")
        self.proc.wait(timeout=5)


# pieces a random endgame is built from: pawns and rooks dominate real endings
ENDGAME_PIECES = (
    [chess.PAWN] * 5 + [chess.ROOK] * 3 + [chess.KNIGHT, chess.BISHOP] * 2 + [chess.QUEEN]
)


def random_endgame(rng: random.Random, max_per_side: int = 4) -> chess.Board:
    """A random legal position with few pieces.

    The self-play games start from the initial position and stop a few plies after the score
    runs away, so simplified positions are rare in the data and the network saturates in them
    (every move in a won K+2R+2P ending scored about +835). Sampling endgames directly is the
    cheapest way to cover that range.
    """
    for _ in range(400):
        board = chess.Board.empty()
        free = list(range(64))
        rng.shuffle(free)
        white_king, black_king = free.pop(), free.pop()
        if chess.square_distance(white_king, black_king) < 2:
            continue
        board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
        for color in (chess.WHITE, chess.BLACK):
            for _ in range(rng.randint(1, max_per_side)):
                piece = rng.choice(ENDGAME_PIECES)
                square = free.pop()
                if piece == chess.PAWN and chess.square_rank(square) in (0, 7):
                    piece = chess.KNIGHT  # a pawn cannot stand on the back ranks
                board.set_piece_at(square, chess.Piece(piece, color))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.clear_stack()
        if board.is_valid() and not board.is_game_over(claim_draw=True):
            return board
    return chess.Board()


def play_game(engine: Teacher, rng: random.Random, depth: int,
              random_plies: int, random_prob: float, max_plies: int,
              start: chess.Board | None = None, lopsided_limit: int = 6) -> list[tuple]:
    board = chess.Board() if start is None else start
    rows: list[tuple] = []
    for _ in range(random_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    ply = 0
    lopsided = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        cp, best_uci = engine.analyse(board.fen(), depth)
        # stop once the game is decided; long won-position tails add little, except when the
        # point of the game is the conversion (endgame starts), which is what we lack
        lopsided = lopsided + 1 if abs(cp) >= 1500 else 0
        if lopsided >= lopsided_limit:
            break
        best = chess.Move.from_uci(best_uci) if best_uci else None
        if best is not None and best not in board.legal_moves:
            best = None
        squares, meta = encode_board(board)
        meta[3] = 1 if board.is_check() else 0
        meta[4] = 1 if (best is not None and board.is_capture(best)) else 0
        rows.append((squares, meta, cp, ply))
        if best is None or rng.random() < random_prob:
            move = rng.choice(list(board.legal_moves))
        else:
            move = best
        board.push(move)
        ply += 1
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        # unfinished games: adjudicate by the last teacher score
        if rows:
            last_cp = rows[-1][2] if board.turn == chess.WHITE else -rows[-1][2]
            result = 1 if last_cp > 400 else (-1 if last_cp < -400 else 0)
        else:
            result = 0
    else:
        result = 1 if outcome.winner == chess.WHITE else -1
    return [(s, m, cp, result, ply) for (s, m, cp, ply) in rows]


def worker(worker_id: int, positions: int, depth: int, seed: int, chunk: int,
           endgame_frac: float = 0.0, tag: str = "chunk") -> None:
    rng = random.Random(seed * 1000 + worker_id)
    engine = Teacher(TEACHER)
    DATA.mkdir(exist_ok=True)
    buf_b: list[np.ndarray] = []
    buf_m: list[np.ndarray] = []
    buf_s: list[int] = []
    buf_r: list[int] = []
    buf_g: list[int] = []
    buf_p: list[int] = []
    total = 0
    game_id = 0
    chunk_id = 0
    started = time.time()
    try:
        while total < positions:
            if rng.random() < endgame_frac:
                # an endgame start: no random opening plies, and let the conversion play out
                rows = play_game(
                    engine, rng, depth, 0, rng.choice([0.0, 0.05]), 160,
                    start=random_endgame(rng), lopsided_limit=40,
                )
            else:
                random_plies = rng.randint(2, 8)
                random_prob = rng.choice([0.0, 0.05, 0.1, 0.2])
                rows = play_game(engine, rng, depth, random_plies, random_prob, 240)
            for squares, meta, cp, result, ply in rows:
                buf_b.append(squares)
                buf_m.append(meta)
                buf_s.append(cp)
                buf_r.append(result)
                buf_g.append(game_id)
                buf_p.append(ply)
            game_id += 1
            total += len(rows)
            if len(buf_s) >= chunk or total >= positions:
                out = DATA / f"{tag}_{worker_id:02d}_{chunk_id:04d}.npz"
                np.savez_compressed(
                    out,
                    boards=np.stack(buf_b), meta=np.stack(buf_m),
                    scores=np.array(buf_s, dtype=np.int16),
                    results=np.array(buf_r, dtype=np.int8),
                    games=np.array(buf_g, dtype=np.int32),
                    plies=np.array(buf_p, dtype=np.int16),
                )
                chunk_id += 1
                buf_b, buf_m, buf_s, buf_r, buf_g, buf_p = [], [], [], [], [], []
                rate = total / max(1e-9, time.time() - started)
                print(f"worker {worker_id}: {total} positions, {game_id} games, "
                      f"{rate:.0f} pos/s", flush=True)
    finally:
        engine.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--positions", type=int, default=250_000, help="per worker")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--chunk", type=int, default=50_000)
    parser.add_argument("--worker-id", type=int, default=None, help="internal")
    parser.add_argument(
        "--endgame-frac",
        type=float,
        default=0.0,
        help="share of games started from a random legal endgame instead of the initial "
        "position, and played until the conversion is over",
    )
    parser.add_argument(
        "--tag", default="chunk", help="chunk filename prefix; use a new one to keep old data"
    )
    args = parser.parse_args()
    if args.worker_id is not None:
        worker(args.worker_id, args.positions, args.depth, args.seed, args.chunk,
               args.endgame_frac, args.tag)
        return
    import subprocess

    procs = []
    for w in range(args.workers):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-id", str(w),
               "--positions", str(args.positions), "--depth", str(args.depth),
               "--seed", str(args.seed), "--chunk", str(args.chunk),
               "--endgame-frac", str(args.endgame_frac), "--tag", args.tag]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
