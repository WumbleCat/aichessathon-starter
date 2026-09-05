"""Cross-check the compiled engine (dc_engine + dc_search) against the python-chess path.

    .venv/Scripts/python.exe agents/29_deepchess/tools/validate_numba.py

Checks: incremental accumulator evaluation equals the from-scratch network evaluation along
random games; hashes stay consistent through make/unmake; the compiled search finds the
same mates as the python search; nodes per second of both engines (CPU time).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
sys.path.insert(0, str(AGENT_DIR))

t0 = time.perf_counter()
c0 = time.process_time()
import agent  # noqa: E402

print(f"agent import returned after {time.perf_counter() - t0:.1f}s wall "
      f"(compiled ready: {agent._compiled_ready()})", flush=True)
if agent._compile_thread is not None:
    agent._compile_thread.join()
import dc_engine  # noqa: E402
import dc_search  # noqa: E402

print(f"import + compile: {time.perf_counter() - t0:.1f}s wall, "
      f"{time.process_time() - c0:.1f}s cpu, engine={agent.ENGINE}, "
      f"error={agent._compile_error!r}", flush=True)
assert agent._numba_searcher is not None
ns = agent._numba_searcher
m = ns.model


def numba_eval_here() -> int:
    return int(dc_search.evaluate_pos(ns._args()))


def check_incremental(seed: int, plies: int) -> tuple[int, int]:
    """Play random moves through make_move; compare eval and hash with python-chess."""
    rng = random.Random(seed)
    board = chess.Board()
    ns.prepare(board, max_nodes=10**9)
    p = ns.pos
    worst = 0
    hash_mismatch = 0
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        code = agent._move_code(board, mv)
        dc_engine.make_move(p.board, p.state, p.hash, code, p.undo, p.acc, m["w1"], m["b1"],
                            dc_engine.ZOBRIST, dc_engine.Z_CASTLE, dc_engine.Z_EP,
                            dc_engine.Z_SIDE, dc_engine.CASTLE_MASK, p.hist)
        board.push(mv)
        assert not dc_engine.left_king_in_check(
            p.board, p.state, dc_engine.KNIGHT_T, dc_engine.KING_T, dc_engine.BISHOP_RAYS,
            dc_engine.ROOK_RAYS, dc_engine.PAWN_ATTACKERS)
        for mode in ("net", "hand"):
            agent.EVAL_MODE = mode
            ns.params[dc_search.P_MODE] = agent._MODE_CODE[mode]
            a = numba_eval_here()
            b = agent.evaluate(board)
            worst = max(worst, abs(a - b))
            if abs(a - b) > 2:
                print(f"  eval mismatch ({mode}) {a} vs {b} at {board.fen()}")
        if int(p.hash[0]) != dc_engine.hash_of_board(board):
            hash_mismatch += 1
        if p.state[dc_engine.S_PLY] >= dc_engine.MAX_PLY - 4:
            break
    agent.EVAL_MODE = "net"
    ns.params[dc_search.P_MODE] = 0
    return worst, hash_mismatch


# move generator: perft against known node counts (castling, promotions, en passant)
PERFT = [
    (chess.STARTING_FEN, 4, 197281),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333),
    ("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 3, 9467),
    ("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379),
]
for fen, depth, expect in PERFT:
    board = chess.Board(fen)
    ns.prepare(board, max_nodes=10**9)
    p = ns.pos
    count = int(dc_engine.perft(
        p.board, p.state, p.hash, p.undo, p.moves, p.acc, m["w1"], m["b1"], dc_engine.ZOBRIST,
        dc_engine.Z_CASTLE, dc_engine.Z_EP, dc_engine.Z_SIDE, dc_engine.CASTLE_MASK, p.hist,
        depth, dc_engine.KNIGHT_T, dc_engine.KING_T, dc_engine.BISHOP_RAYS, dc_engine.ROOK_RAYS,
        dc_engine.PAWN_ATTACKERS))
    print(f"perft {fen[:24]:24} d{depth}: {count} {'ok' if count == expect else 'MISMATCH ' + str(expect)}",
          flush=True)

worst_total = 0
hash_bad = 0
for seed in range(12):
    w, h = check_incremental(seed, 80)
    worst_total = max(worst_total, w)
    hash_bad += h
print(f"incremental eval: worst |diff| = {worst_total} cp, hash mismatches = {hash_bad}",
      flush=True)

MATES = [
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
    ("8/4P3/3k4/1P6/1PP5/7B/8/K3R3 w - - 0 1", "e7e8n"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", None),  # must not stalemate
]
agent.ENGINE = "numba"
for fen, expect in MATES:
    board = chess.Board(fen)
    uci = agent.get_move(fen, 3000)
    mv = chess.Move.from_uci(uci)
    assert mv in board.legal_moves, (fen, uci)
    board.push(mv)
    if expect is not None:
        status = "ok" if uci == expect else "WRONG"
    else:
        status = "ok" if not board.is_stalemate() else "STALEMATE"
    print(f"mate test {fen[:30]:30} -> {uci} depth {agent.STATS['depth']} "
          f"score {agent.STATS['score']} {status}", flush=True)

# speed: fixed node budget, CPU time
fen = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8"
board = chess.Board(fen)
ns.new_game()
ns.prepare(board, max_nodes=300_000)
root = [(agent._move_code(board, mv), mv) for mv in board.legal_moves]
c0 = time.process_time()
t0 = time.perf_counter()
depth = 0
try:
    for d in range(1, 40):
        score, best, root = ns.search_root(d, root)
        depth = d
        print(f"  numba depth {d}: {best.uci()} {score} nodes {int(ns.stats[0])}", flush=True)
except agent.OutOfTime:
    pass
cpu = time.process_time() - c0
wall = time.perf_counter() - t0
nodes = int(ns.stats[dc_search.ST_NODES])
print(f"numba search: depth {depth}, {nodes} nodes, {cpu:.2f}s cpu -> {nodes / cpu:,.0f} nps "
      f"(wall {wall:.1f}s)", flush=True)


class NodeLimited(agent.Searcher):
    max_nodes = 30_000

    def _check_time(self) -> None:
        if self.nodes >= self.max_nodes:
            raise agent.OutOfTime()


s = NodeLimited()
s.armed = True
c0 = time.process_time()
ordered = list(board.legal_moves)
best = None
depth = 0
try:
    for d in range(1, 40):
        sc, best, ordered = s.search_root(board, d, ordered, best)
        depth = d
except agent.OutOfTime:
    pass
cpu = time.process_time() - c0
print(f"python search: depth {depth}, {s.nodes} nodes, {cpu:.2f}s cpu -> {s.nodes / cpu:,.0f} nps",
      flush=True)
print("done")
