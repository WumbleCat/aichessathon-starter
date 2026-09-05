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
| rook ending | 8 | 53 % |
| pawn ending | 0 | 40 % |

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

See RESULTS.md for the shipped model's numbers and the table below for this session's.
