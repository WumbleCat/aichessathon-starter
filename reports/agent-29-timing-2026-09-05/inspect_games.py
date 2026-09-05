"""Reconstruct the games recorded by move_timing.py from the per-move FENs."""

import json
import sys

import chess

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]

# split into games: a new game starts when the fullmove number drops
games: list[list[dict]] = []
last_ply = -1
for r in records:
    board = chess.Board(r["fen"])
    ply = board.fullmove_number * 2 + (0 if board.turn == chess.WHITE else 1)
    if ply <= last_ply and ply < 4:
        games.append([])
    if not games:
        games.append([])
    games[-1].append(r)
    last_ply = ply

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board: chess.Board) -> tuple[int, int]:
    w = sum(v * len(board.pieces(p, chess.WHITE)) for p, v in VAL.items())
    b = sum(v * len(board.pieces(p, chess.BLACK)) for p, v in VAL.items())
    return w, b


for gi, game in enumerate(games):
    print(f"=== game {gi + 1}: {len(game)} move requests")
    prev = None
    for r in game:
        board = chess.Board(r["fen"])
        w, b = material(board)
        san = ""
        if prev is not None:
            # find the move that led from prev to this position
            for m in prev.legal_moves:
                prev.push(m)
                same = prev.board_fen() == board.board_fen() and prev.turn == board.turn
                prev.pop()
                if same:
                    san = prev.san(m)
                    break
        tag = r["agent"]
        print(f"{board.fullmove_number:3d}{'.' if board.turn else '...':3s} last={san:7s} "
              f"to-move={tag:8s} mat W{w:2d} B{b:2d} left={r['time_left_ms']:6d} "
              f"took={r['move_ms']:7.1f}")
        prev = board
    print("final fen:", game[-1]["fen"])
