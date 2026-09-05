# Agent 29 (DeepChess pairwise engine): harness runs and move timing, 2026-09-05

Run by a background session at the request of the interactive session that owns
`agents/29_deepchess`. Nothing under that directory was edited; the agent under test is the
working tree of the shared checkout with `agent.py` last modified 09:50:57 (games before
09:51 used the previous save, see `mtimes_at_arena_start.txt`). All runs are on the shared
dev machine under its usual load (dozens of Python processes), one game at a time.

## Harness arena (10 s + 0.1 s)

| opponent | games | +W =D -L | terminations |
|---|---|---|---|
| baselines/numba | 6 | +5 =0 -1 | flag 5 (all the opponent's), checkmate 1 (our loss as Black in game 2) |
| baselines/random | 4 | +3 =1 -0 | checkmate 3, threefold_repetition 1 |

No init, flag, crash or illegal move by our agent. `harness.arena` still exits 1 on the numba
run because it counts any flag termination, including the opponent's.

## Per-move wall time at the real clock (120 s + 0.5 s)

Measured through the unmodified harness referee with `move_timing.py` (a subclass of the
harness `Agent` that times each `move` call exactly as the referee does).

| opponent | games | +W =D -L | our moves | median | p90 | p99 | max | max clock fraction | init |
|---|---|---|---|---|---|---|---|---|---|
| baselines/greedy | 2 | +0 =2 -0 | 80 | 2726 ms | 3253 ms | 3566 ms | 3968 ms | 7.4 % | 77.0 s, 70.8 s |
| baselines/minimax | 2 | +0 =0 -2 | 62 | 2849 ms | 3364 ms | 5250 ms | 5714 ms | 12.7 % | 72.7 s, 70.8 s |

Clock handling is safe: no move above 6 s, never more than 13 % of the remaining clock.

## Finding: one-ply play for the whole game while the compile thread runs

The results above are not noise. Reconstructing the games from the recorded FENs
(`inspect_games.py`): as White against greedy the agent played 4.Qxg7?? Bxg7 and ended with a
bare king (draw by insufficient material, greedy had K+P); as Black it won K+Q+B+4P against a
bare king and then shuffled Qb6+/Qf6/Qe6 into a threefold. Against minimax it allowed
5...Qxg2 and 6...Qxh1+ as White and was mated in both games.

`probe_compile_window.py` imports the agent as the runner does and calls `get_move` at a
120 s clock in a loop until the compile thread ends (`probe_compile_window.log`):

- import returned after 70.3 s wall (16.2 s CPU) with the compile thread still alive;
- for the next 94 s of wall time every call returned depth 0 with nodes=1 after spending its
  full budget (3.3 s on the start position, 3.5 s on a position where Qxg7 hangs the queen,
  6.3-7.4 s on K+Q+B vs K), always choosing the one-ply static pick, including the queen hang;
- the first call after the thread ended reached depth 4 on K+Q+B vs K (mate found), then
  depth 10 / 360k nodes on the start position and depth 7 / 699k nodes on the Qxg7 position.

So while numba compiles, the pure-Python fallback gets essentially no GIL time (one node in
three seconds), runs out its deadline, and falls through to `_quick_move`. A 40-move game at
3 s per move fits inside the window on this machine, which is why the full-clock games were
one-ply games while the 10 s arena games, which mostly outlive the window, looked fine.

On the platform's idle dedicated core the bounded join at import (70 s budget against a
~30 s CPU compile) should finish before the first move, so this is most likely a local
artefact. Two cheap hardenings if a slow platform core is a concern: when `_compiling()` is
true at a long clock, join the compile thread for a bounded slice of the first move; and
while compiling answer with the one-ply pick at once instead of burning 3-7 s to reach the
same move.

## Files

- `arena_numba_6.log`, `arena_random_4.log`: harness arena output.
- `timing29_greedy.jsonl`, `timing29_minimax.jsonl`: one record per move request (both
  sides) with FEN, clock, wall time and clock fraction; `*.summary.json` the aggregates.
- `probe_compile_window.log`: the depth trace above.
- `move_timing.py`, `probe_compile_window.py`, `inspect_games.py`: the scripts, runnable with
  the repository venv from anywhere (they locate the repo by absolute path).
