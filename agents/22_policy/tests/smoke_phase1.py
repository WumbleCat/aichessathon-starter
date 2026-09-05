"""Phase 1 smoke test: import succeeds and get_move returns a legal move."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402

import agent  # noqa: E402

board = chess.Board()
uci = agent.get_move(board.fen(), 1000)
print(uci, chess.Move.from_uci(uci) in board.legal_moves)
