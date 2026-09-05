# DeepChess: strength work, 2026-09-05 (session 31)

Changes made to raise the win rate against a strong engine, and what each is worth. The
engine, the model and the harness are unchanged unless stated; every change is behind an
environment flag so it can be measured against the same build rather than a remembered
number.

## Target, stated honestly

The brief was "always beat Stockfish Elo 2600". *Always* is not reachable and neither is
"perfection": chess is drawish at the top, so even an engine 400 points stronger than its
opponent draws a fair share of games, and a 207k-parameter evaluator searching ~200k
nodes/s on one core is not going to be 400 points above 2600. What follows raises the win
rate as far as the evidence supports and reports the number honestly at each step.

## How things are measured

`tools/engine_bench.py` with paired openings (each of 42 positions played twice, colours
swapped), Stockfish held at a fixed `UCI_Elo` with 100 ms a move, and the agent on a 70 s
clock with no increment, which makes a complete game last about 80 seconds. The A/B variant
in `variants/base/` loads this same `agent.py` with the three flags off, so the two arms
differ only in the changes and share the machine's load minute by minute (the tool
interleaves them).

## Changes

### 1. Aspiration windows at the root (`DEEPCHESS_ASPIRATION`, default on)

The root previously searched every iteration with a full window. It now searches inside
`prev_score ± 30` and widens by 4x on a fail, re-using the improved move order from the
failed attempt. Measured on six fixed positions at a 8 s clock the depth reached was 49 vs
48 plies summed, i.e. within noise; the value shows up in games rather than in fixed-position
depth, so it is carried by the A/B result below and not claimed on its own.

### 2. Phase-adaptive evaluation blend (`DEEPCHESS_ADAPTIVE_BLEND`, default on)

The shipped evaluation is `0.75 * network + 0.25 * handcrafted`. The known weakness of the
network is saturation outside its training range: every move in a won K+2R+2P ending scores
about +835, so it cannot tell a conversion from a shuffle. The blend is now a function of the
game phase, from 80 % network with all the pieces on down to 40 % in a pawn ending, where the
handcrafted material and king-activity terms still mean something. One value per search,
since the phase barely moves inside a tree, so it costs nothing in the hot loop.

| position | phase | network share |
|---|---|---|
| initial / middlegame | 24 | 80 % |
| rook ending | 8 | 70 % |
| pawn ending | 0 | 65 % |

The endgame endpoint is 0.65 and not lower for a reason worth recording. At 0.40 the agent
answered K+P vs K (`8/1P4k1/8/8/8/8/8/K7 w`) with a king move instead of promoting until it
could search 10 plies, and `tests/test_agent.py::test_queen_promotion` caught it. Both moves
win, but shuffling instead of promoting is how won endings turn into fifty-move draws, and it
is the opposite of what this change was for. The cause is that the handcrafted evaluation has
its own endgame bias toward king activity, so pulling too much weight away from the network
just swaps one distortion for another. At 0.65 the engine promotes at every clock from 500 ms
up while the network's saturation is still damped.

### 3. Time given to an unsettled root (`DEEPCHESS_EXTEND_UNSTABLE`, default on)

The iteration loop stopped as soon as the elapsed time passed the soft budget. It now allows
35 % beyond it when the best move changed at the last completed depth, and 60 % when the score
fell by more than 30 cp, decaying back once the root settles. The hard budget is unchanged, so
this spends the move's own reserve and cannot cost the game's clock.

## Training work

`training/gen_data.py` gained `--endgame-frac` and `--tag`. The self-play games all start
from the initial position and stop six plies after the score runs away, so simplified
positions are rare in the 1M-position dataset, which is exactly why the network saturates
there. With `--endgame-frac` a share of games start from a random legal endgame (2 to 5 men a
side, validated with python-chess) and are played until the conversion is over
(`lopsided_limit` 40 rather than 6). `--tag` keeps the new chunks beside the old ones instead
of overwriting them.

`training/train.py` gained `--init`, which fine-tunes from an existing `.pt` checkpoint
instead of starting from random weights.

## Results

All matches: paired openings from `agents/31_tdleaf/training/openings.txt` (42 positions,
each played twice with the colours swapped), Stockfish at a fixed `UCI_Elo` with 100 ms a
move, agent on a 70 s clock with no increment, which makes a game last about 80 s. The two
arms of each A/B are interleaved by the runner, so they share the machine's load.

| test | arm | games | +W =D -L | score |
|---|---|---|---|---|
| search/eval changes, vs SF 2200 | with (flags on) | 40 | +26 =6 -8 | **72.5 %** |
| | without (`variants/base`) | 40 | +21 =10 -9 | 65.0 % |
| search/eval changes, vs SF 2600 | with, blend endpoint 0.65 | 24 | +10 =8 -6 | **58.3 %** |
| | without (`variants/base`) | 24 | +7 =8 -9 | 45.8 % |
| v2 network, vs SF 2200 | v2 (shipped) | 24 | +19 =5 -0 | **89.6 %** |
| | v1 (`variants/v1model`) | 24 | +17 =1 -6 | 72.9 % |
| v3 network, vs SF 2600 | v3 | 24 | +12 =5 -7 | 60.4 % |
| | v2 (`variants/v2model`) | 24 | +10 =9 -5 | 60.4 % |

**v3 is not shipped.** A second round added 600k more positions (2M raw, 1.4M kept) and 22
epochs, and every held-out metric improved again: pairwise accuracy 0.9421 and MAE 79.4 cp
against v2's 0.9408 and 85.8 cp. Over 24 games a side against Stockfish 2600 the two scored
*exactly* the same. Its first 15 games read 70 % against 56.7 %, which would have been a
tempting place to stop; it was noise. Better validation numbers did not become strength, so
`models/deepchess.npz` stays v2, the version with a measured win, and v3 is kept beside it.

The v2 network is the larger of the two effects and it lost no game in 24. Both arms of that
test run the same search with the same flags, so the difference is the weights alone.

### Against Stockfish at UCI_Elo 2600

| opponent's thinking time | games | +W =D -L | score | implied Elo |
|---|---|---|---|---|
| 100 ms a move (the standard benchmark setting) | 40 | +21 =9 -10 | 63.7 % | 2698 ± 128 |
| a thirtieth of its clock, ~2.3 s, the same as the agent | 24 | +9 =7 -8 | **52.1 %** | 2614 ± 150 |

The second row is the honest one and the one to quote: both sides think for about the same
time per move, and no game in either row ended in a failure by either side. The agent is
level with Stockfish held at 2600, not above it, and it loses roughly a third of the games.

**"Always beat 2600" is not what this shows and is not reachable.** A third of the games are
losses and another quarter are draws. Chess is drawish between near-equal opponents, and a
207k-parameter evaluator on one core does not become deterministic against a 2600-strength
opponent by training harder. Note also that Stockfish's `UCI_Elo` is its own nominal scale,
not a FIDE rating, so read these as a consistent yardstick rather than an absolute rating.

### An artefact that had to be fixed first

The first equal-time attempt gave 80 % and was wrong. Stockfish was asking for a fixed 1500 ms
a move while the referee charged it wall time from a 70 s clock, so it flagged in 12 of 20
games and every one was scored as an agent win. `tools/engine_bench.py` now caps the engine
at a thirtieth of its remaining clock and records which side failed, so an engine flag can
never again be read as agent strength. The 32-agent sweep was unaffected: 100 ms a move
against a 10 s clock never approaches the budget.

Held-out metrics of the fine-tune (`train_v2_gpu.log`), 1.4M positions, 8 epochs, lr 3e-4:

| metric | v1 | v2 |
|---|---|---|
| pairwise accuracy | 0.9290 | **0.9408** |
| Spearman vs teacher | 0.9543 | **0.9643** |
| MAE after calibration | 96.6 cp | **85.8 cp** |

## Training on the GPU

`train.py --device auto|cpu|cuda`. The same 8-epoch fine-tune takes 196 s on the CPU and
24 s on the RTX 3060, with the metrics unchanged.

The GPU run needs a CUDA build of torch, and the repository's `.venv` is deliberately pinned
to `2.13.0+cpu` (it mirrors the platform, which has no GPU) and is shared by every other
agent session, so replacing torch there would be both wrong and disruptive. Training uses a
separate `.venv-cuda` with `torch==2.13.0+cu126` instead:

```powershell
C:\Python314\python.exe -m venv .venv-cuda
.venv-cuda\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0+cu126 numpy
.venv-cuda\Scripts\python.exe training/train.py --data "data/*.npz" --init models/deepchess_v1.pt --out models/deepchess_v2.npz --epochs 8 --lr 3e-4 --device cuda
```

The agent, the tests and the harness keep using `.venv`, and the shipped weights are a
device-independent `.npz`, so nothing about the submission depends on the GPU. `.venv-cuda`
is git-ignored.
