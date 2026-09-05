# Results: Giraffe-style learned evaluation (agent 30)

All measurements were taken on a shared Windows machine running 70-90 other Python
processes (other agents' training and arenas), so wall-clock figures are pessimistic by a
factor of 3-10 and vary between runs. Fixed-depth experiments are the ones to trust.

## The controlled experiment: same search, two evaluators

`training/selfplay_arena.py` plays paired games (both colours from the same random
6-ply opening) with the identical search and only the leaf evaluator swapped. Fixed
depth 3, no clock.

| evaluator A | evaluator B | games | A result | A score | Elo (±) | file |
|---|---|---|---|---|---|---|
| absolute net (bootstrap on quiescence-resolved static labels) | static (hce) | 24 | +4 =0 -20 | 16.7% | -280 (95) | `results/selfplay_net_vs_hce_depth3.txt` |
| static + residual net, 14k-position partial training set | static (hce) | 16 | +5 =1 -10 | 34.4% | -112 (91) | (scratch run) |
| static + residual net, 158k-position training set | static (hce) | 20 | +12 =3 -5 | 67.5% | +127 (83) | `results/selfplay_residual_vs_hce_depth3.txt` |

The shipped `models/giraffe.npz` is the same recipe as the last row retrained on the full
171,890-position set for 30 epochs (held-out residual error 33 cp instead of 36 cp); the
arena row was played with the 158k/20-epoch checkpoint because the full relabel was still
running.

Reading: a network trained to *replace* the static score was clearly harmful. On 400
probe positions its mean absolute error against its own label was 103 cp on quiet
positions (p90 317 cp) and 162 cp on tactical ones; a search that trusts an evaluator
which misjudges material by a pawn loses to one that counts it exactly. Keeping the
static score exact and learning only the residual towards a depth-2 search score turns
the same network into a positive contribution.

## Offline proxy for the residual net

Held out 10% of the quiet positions (`training/proxy_eval.py`, 158k snapshot):

| quantity | value |
|---|---|
| residual `depth-2 score - static`, mean / std / mean abs / p90 abs | +43 / 209 / 88 / 298 cp |
| error of static alone against the depth-2 score (mean abs) | 87 cp |
| error of static + network (mean abs) | 36 cp |
| correlation of predicted and true residual | 0.92 |
| best shrinkage factor on the residual | 1.00 |

Full-set training (`results/bootstrap_residual.log`): best validation 33 cp mean abs
error after 28 of 30 epochs, 1e-4 weight decay, train/val MSE 0.0062 / 0.0089 in tanh
space, about 6 s per epoch with one torch thread.

## Data

- `training/gen_positions.py`: 240,000 positions from noisy one-ply self-play, 920 s on
  6 workers (`results/gen_positions.log`).
- `training/relabel.py --depth 2`: 171,890 quiet positions kept (not in check, quiescence
  value equal to static), labelled by a depth-2 alpha-beta search of the handcrafted
  evaluator, 10,686 s on 4 starved workers (`results/relabel_d2.log`).
- No third-party engine, network or data is involved anywhere.

## Speed

`bench.py --budget 1.5` at the start of the session (machine loaded, but less than later):

| evaluator | latency per position | nodes/s | mean depth over 8 positions |
|---|---|---|---|
| static + residual net | 35 us | ~9,500 | 4.2 |
| static only | 4.7 us | ~7,200 | 3.8 |

The evaluator is roughly a third of the search time; the python-chess move generation
and the Python search loop dominate. Network: 48,705 parameters, 191 KB on disk. Import
plus numba compilation: 20 s cold on the loaded machine, under a second with a warm
numba cache, both inside the 90 s budget.

## Harness games

`python -m harness.arena --agent agents/30_giraffe --opponent baselines/minimax --games 6`
with the shipped agent and the real clock (`results/arena_net_vs_minimax.txt`): 6-0 for
the agent, but every game ended on the opponent's flag because the machine was saturated
(the minimax baseline does not manage its clock under load), so this says nothing about
strength. The earlier static-evaluator runs (`results/arena_hce_vs_*.txt`) show the same
load pattern with `init` and `flag` terminations on both sides. Real-clock baseline
games have to be replayed on a quiet machine before the numbers mean anything; the agent
itself never flagged in this session's harness run.

## Correctness

`python -m unittest discover -s tests`: 30 tests (move types, castling, promotion, en
passant, check evasion, mate/stalemate, clocks down to 50 ms, repeated calls, feature
symmetry, torch/numba parity on the residual) pass.

## What was not done

- TD-Leaf (stage 3) runs end to end (`training/tdleaf.py`, smoke-tested) but no TD-Leaf
  checkpoint was trained long enough to gate; the shipped net is the supervised residual.
- Hundreds of paired games against the baselines were not affordable on the shared
  machine; the evidence for the learned evaluator is the fixed-depth controlled arena
  and the held-out proxy above.
