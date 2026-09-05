"""How deep does agents/29_deepchess search while its compile thread is still running?

Imports the agent exactly as the runner does (its directory first on sys.path), then calls
get_move at the real clock repeatedly and prints, for each call, whether the compile thread
is still alive, the depth reached, nodes, and wall time, until the compiled engine is ready
(or a wall-time cap is hit). Nothing under the agent directory is written by this script.
"""

import os
import sys
import time

AGENT_DIR = "E:/sourcecode/ai-chess-original/aichessathon-starter/agents/29_deepchess"
sys.path.insert(0, AGENT_DIR)
os.chdir(AGENT_DIR)

t_import = time.perf_counter()
c_import = time.process_time()
import agent  # noqa: E402

print(f"import returned after wall {time.perf_counter() - t_import:.1f}s "
      f"cpu {time.process_time() - c_import:.1f}s; compiling={agent._compiling()} "
      f"engine={agent.ENGINE}", flush=True)

FENS = [
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("queen hangs on g7 after Qxg7?", "r1b1kbnr/pppp1ppp/2n5/p7/3QP3/8/PPP2PPP/RNB1KBNR w KQkq - 0 4"),
    ("KQB vs K", "8/ppp2p2/4q3/1K1kb3/8/8/8/8 b - - 23 41"),
]

cap = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
t0 = time.perf_counter()
call = 0
while True:
    for name, fen in FENS:
        call += 1
        compiling = agent._compiling()
        t1 = time.perf_counter()
        move = agent.get_move(fen, 120_000)
        wall = time.perf_counter() - t1
        s = agent.STATS
        print(f"[{time.perf_counter() - t0:6.1f}s] compiling={compiling!s:5} {name:32s} "
              f"move={move:6s} depth={s.get('depth')} nodes={s.get('nodes')} "
              f"took={wall * 1000:.0f}ms", flush=True)
    if not agent._compiling() or time.perf_counter() - t0 > cap:
        break

print(f"compile thread alive={agent._compiling()} after {time.perf_counter() - t0:.0f}s; "
      f"process cpu {time.process_time():.1f}s", flush=True)
if not agent._compiling():
    for name, fen in FENS:
        t1 = time.perf_counter()
        move = agent.get_move(fen, 120_000)
        s = agent.STATS
        print(f"compiled engine: {name:32s} move={move:6s} depth={s.get('depth')} "
              f"nodes={s.get('nodes')} took={(time.perf_counter() - t1) * 1000:.0f}ms",
              flush=True)
