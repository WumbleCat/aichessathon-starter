# round robin results

`games.jsonl` is the live record: one JSON object per finished game, appended as it finishes.
`round_robin.py --report` turns it into standings, an Elo fit with error bars, and a cross table.
The supervisor also writes `standings_latest.txt` every thousand games.

## Why the first run was thrown away

`games_contaminated_14workers_*.jsonl` holds 1086 games played at 14 concurrent games. At that
concurrency the three agents that compile a numba search at import (18_alpha_beta, 20_pvs,
21_nnue) never finished compiling and played a pure Python fallback for whole games. Measured
with `agent_health.py`, with the tournament stopped:

| agent | import | median move | compiled search |
| --- | --- | --- | --- |
| 20_pvs | 39.1 s | 0.61 s | yes |
| 21_nnue | 66.4 s | 0.53 s | yes, `nnue=yes` |

and with 14 games running alongside:

| agent | import | median move | compiled search |
| --- | --- | --- | --- |
| 20_pvs | 61.1 s | 0.00 s | no, moves instantly |
| 21_nnue | 71.0 s | 0.30 s | no, "still compiling" |

So those games measured the machine rather than the agents. The rerun caps how many games with a
slow agent may import at once (`--compile-slots`) near the number of free cores, which lets the
jit land inside the 90 s import budget. The old file is kept because the games themselves were
played honestly; it just answers a different question.

## Reading this run

The live run is 12 concurrent games with 6 compile slots. Lowering concurrency further did not
help: at 8 workers the three slow agents still failed to compile, and the run slowed to 38 hours,
so the low setting was buying nothing. The spare capacity on this machine belongs to other
sessions and moves between 1 and 7 free cores, which is why the handicap comes and goes.

Read the score column together with `round_robin.py --health`:

    python round_robin.py --health              # import and median move time, per agent
    python round_robin.py --report              # standings over every game
    python round_robin.py --report --min-move-s 0.05   # only games both sides searched

A median move time near zero means the agent returned without searching, so its games say
nothing about its strength. As of the first 150 games of this run, 20_pvs sat at 0.00 s in every
game, 21_nnue at 0.28 s against 0.53 s when healthy, and 18_alpha_beta at 0.22 s. Treat the
placings of those three as a lower bound, and the other fourteen as sound.
