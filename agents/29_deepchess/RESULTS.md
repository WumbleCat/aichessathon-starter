# DeepChess pairwise engine: results

All numbers from 2026-09-05 on the shared dev machine (16 cores, 60-90 concurrent Python
processes from other sessions, 2 GB free RAM). Wall-clock there is 8-25x CPU time, so speed
is reported in CPU seconds and matches are node- or CPU-limited (`tools/selfplay.py`).

## Model

`models/deepchess.npz` = v1 (`train_v1.log`, `models/deepchess_v1.json`).

| item | value |
|---|---|
| architecture | 773 binary features -> 256 -> 32 -> 32 -> 1, clipped ReLU |
| parameters | 207,458 |
| training data | 697,508 quiet positions (no check, no capture as best move) from 22,228 self-play games labelled by Stockfish at depth 8; scores clipped to 2000 cp |
| objective | pairwise BCE on `(V(A) - V(B)) / tau` for in-batch pairs with a 30 cp margin, plus 0.25 x SmoothL1 value regression |
| epochs / batch / lr | 20 / 8192 / 1e-3 one-cycle |
| validation (36k positions, held-out games) | pairwise accuracy 0.929, Spearman 0.954 vs teacher, MAE 97 cp after linear calibration |
| sanity | identical positions give p = 0.5, swapping A/B inverts p exactly (`tests/test_agent.py`) |

Known weakness: outside the training range the network saturates (every move in a
K+2R+2P vs K position scores +835), so won endgames are converted by the mate search
rather than by the evaluation. The `blend` mode mixes in the handcrafted PST evaluation
for exactly this case (see self-play below).

## Engine speed

`tools/validate_numba.py`, middlegame position, single process, CPU time:

| engine | nodes/s (cpu) | depth reached at 300k nodes |
|---|---|---|
| compiled, first version | 81,700 | 7 |
| compiled, hidden layers stored transposed | 93,200 | 7 |
| compiled, plus `fastmath` on the evaluation (vectorised sums) | 208,700 | 7 |
| python-chess search + numba leaf eval | 13,900 | 5 (at 30k nodes) |

The compiled engine evaluates ~15x more nodes per CPU second. The evaluation is still
bit-identical to the python path after both changes (`tools/validate_numba.py`). Both engines find the test
mates (mate in 1, knight-promotion mate, scholar's mate, stalemate avoidance).
`tests/test_agent.py`: 28/28 pass on the current build (legality in castling, promotion,
en passant, check evasion, mate and stalemate positions; clocks 50 ms to 120 s never use
more than 45 % of the clock; repeated calls keep state valid; model sanity checks). Under
the load of this session the compiled engine reached depth 2 on a 500 ms clock, depth 4
on 5 s, depth 7 on 30 s and depth 10 on 120 s (7.5 s used).

## Import / compile budget

`tools/compile_timing.py`, CPU seconds under the machine load above (an idle core is
roughly twice as fast):

| stage | first version | now |
|---|---|---|
| numba import | 2.0 | 1.5 |
| dc_engine (board, movegen, make/unmake) | 22.6 | 15.7 |
| dc_search without `search` | 16 | 23.6 |
| `search` | 85.4 (compiled 3x) | 18.6 |
| total | 126 | ~60 |

The agent compiles in a thread and joins it for what remains of a 70 s budget from the
start of the import; the python engine plays until the compiled one is ready, so init can
never be lost. Where the agent directory is writable (local runs; the platform's
filesystem is read-only) numba's disk cache is on for all kernels: the second process
imported in 11.2 s wall / 2.6 s CPU with the compiled engine ready at import, all
validation checks identical, no crash (`results/validate_cache1.log`, `validate_cache2.log`).

## Matches

Paired openings (6 random plies, handcrafted eval within 80 cp), colours swapped, draws
by threefold/50-move/insufficient material, 300 plies to material adjudication.

| A | B | budget per move | games | A: +W =D -L | A score | notes |
|---|---|---|---|---|---|---|
| numba:net | python:net | 300 ms CPU | 20 | +16 =4 -0 | 90 % | same evaluation, compiled search reaches depth 5.2 on average vs 3.1; ~+380 Elo |
| numba:net | numba:hand | 20,000 nodes | 30 | +14 =7 -9 | 58 % | DeepChess scalar vs material + PST at equal nodes: ~+60 Elo (95 % interval +-130) |
| numba:net | numba:hand | 300 ms CPU | 30 | +6 =7 -17 | 32 % | before the evaluation speedup (93k nodes/s): the cheaper PST eval searched 1.7 plies deeper (7.4 vs 5.7) and won |
| numba:net | numba:hand | 300 ms CPU | 30 | +15 =5 -10 | 58 % | after the speedup (209k nodes/s): depth 6.5 vs 7.2 and the network's knowledge now wins out, ~+60 Elo (+-130) |
| numba:blend:0.75 | numba:net | 20,000 nodes | 30 | +14 =9 -7 | 62 % | 75 % network + 25 % PST fixes the saturation in won positions: ~+80 Elo (+-135); now the default |
| numba:blend:0.75 | numba:hand | 300 ms CPU | 13 (stopped) | +3 =4 -6 | 38 % | before the speedup, stopped when the faster evaluation landed |
| numba:blend:0.75 | numba:hand | 300 ms CPU | 30 | +12 =11 -7 | 58 % | after the speedup: same score as net with fewer losses (blend is the shipped default) |

Harness games (`harness.arena`, 10 s + 0.1 s, protocol and robustness only under this
load; the compiled engine is usually not ready for the first moves and the python engine
plays them):

| opponent | games | +W =D -L | terminations |
|---|---|---|---|
| baselines/greedy (before the compile thread) | 4 | +2 =0 -2 | 2 losses "by init": cold numba compile > 90 s wall under load |
| baselines/greedy | 4 | +3 =1 -0 | checkmate 3, threefold 1; no init, flag, crash or illegal move |
| baselines/minimax | 4 | +4 =0 -0 | all four by the opponent flagging under the load; our agent never flagged |
| baselines/numba (run by the resume session) | 6 | +5 =0 -1 | five opponent flags, one checkmate loss as Black in game 2, most likely inside the compile window where the python engine plays with a quarter budget |
| baselines/random (run by the resume session) | 4 | +3 =1 -0 | checkmate 3, one threefold repetition: the fallback engine shuffling during the compile window; no init, flag, crash or illegal move |
| baselines/minimax at 120 s + 0.5 s, after the compile-window fix and with the engine cache | 2 | +2 =0 -0 | both by checkmate; the same pair before the fix was 0-2 |

## Per-move time at the real clock (README item 6)

Measured by the resume session through the unmodified referee at 120 s + 0.5 s, one game
at a time, under the usual load (agent.py of 09:50, before the compile-window fix below;
scripts, per-move records and the probe trace are on branch `worktree-agent-29-bench`
under `reports/agent-29-timing-2026-09-05/`):

| opponent | games | our moves | median | p90 | p99 | max | max share of clock | init |
|---|---|---|---|---|---|---|---|---|
| greedy | 2 (=2) | 80 | 2.73 s | 3.25 s | 3.57 s | 3.97 s | 7.4 % | 77.0 s, 70.8 s |
| minimax | 2 (-2) | 62 | 2.85 s | 3.36 s | 5.25 s | 5.71 s | 12.7 % | 72.7 s, 70.8 s |

No flag, illegal move or crash. The results themselves were bad: under this load the
compile thread outlives a whole 4-5 minute game, the compiler holds the GIL almost
continuously (a probe saw one search node in 3 s), and a bug then made every move fall
through to the one-ply static pick after spending its budget (a queen hang, a K+Q+B vs K
draw). Since fixed: while compiling, `get_move` waits on the compile thread for the hard
budget (20 s on a fresh clock) and only then answers with the static pick; the compiled
engine takes over the next move. On the platform's idle core the compile is expected to
finish inside the import join, so this window should not exist there.

## Submission check

`harness.package --include models` from the agent directory: 7 files, 2,601,349 bytes
unzipped (limit 50,000,000): `agent.py`, `dc_engine.py`, `dc_search.py`, the weights and
the training record. The extracted zip imported from its own directory (no repository on
the path) returns from import after 70 s wall (11 s CPU) with the python engine playing
legal moves while the compile thread finishes, exactly as designed for the platform's
90 s budget. Only preinstalled packages are imported: python-chess, numpy, numba.
