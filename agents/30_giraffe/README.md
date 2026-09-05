# 30 — Giraffe-style learned evaluation

A chess agent for AI Chessathon built around the central idea of Matthew Lai's *Giraffe*
(2015): learn the evaluation function instead of hand-writing it, then drive an alpha-beta
search with it. The network here is trained from random initialisation by this project;
no third-party engine code or weights are used or shipped.

```
agents/30_giraffe/
  agent.py            get_move(fen, time_left_ms) -> uci; loads weights, time management
  giraffe_eval.py     numba feature extractor, network forward pass, handcrafted control eval
  giraffe_search.py   iterative-deepening PVS with TT, ordering, NMP, LMR, quiescence
  models/giraffe.npz  the shipped flat float32 weight vector (48,705 parameters, ~190 KB)
  training/           model.py (torch twin), gen_positions.py, relabel.py, bootstrap.py,
                      tdleaf.py, selfplay_arena.py (same search, two evaluators)
  tests/              rules/legality/clock/symmetry/parity tests
  bench.py            latency, nodes/s, depth
  IMPLEMENTATION.md   plan;  RESULTS.md  measurements
```

## How it works

**Features (335 floats).** The board is normalised so the side to move is always "us"
(vertical flip plus colour swap for black), which makes the network colour-symmetric and
gives the search a side-to-move score directly. Three groups follow the paper:

| group | size | content |
|---|---|---|
| global | 15 | castling rights (us/them), piece counts (us/them), en passant flag |
| piece-centric | 32 x 6 | fixed slots (K, Q, R, R, B, B, N, N, P x 8 per side): present, file, rank, lowest-valued enemy attacker, lowest-valued defender, mobility |
| square-centric | 64 x 2 | lowest-valued attacker of every square by us and by them |

Everything is computed in one numba pass over the bitboards a `chess.Board` already holds.

**Network.** Giraffe's first hidden layer is split per feature group (16 / 128 / 64 units),
then merged through 64 and 32 ReLU units into a `tanh` output scaled to centipawns
(`600 * tanh(z)`). The forward pass is a numba loop over a flat weight vector, so a
board-to-score evaluation costs roughly 30-50 us on one core.

**Residual design.** The evaluation the search sees is `static + network`, where `static`
is the handcrafted material + piece-square score computed exactly, and the network only
predicts the *residual*: what a deeper search knows about a quiet position that the static
score does not. This is the adaptation of Giraffe's "learn the evaluation" idea that
survived the controlled experiment. The first version, a network trained to replace the
static score outright (labels from a quiescence-resolved static evaluation), had a mean
error of about 100 cp even on quiet positions and lost 4-20 to the static score with the
identical search. Keeping material exact and learning only the correction removes that
failure mode; whether the correction adds strength is then a clean A/B test.

**Search.** Plain python-chess move generation, iterative deepening with principal
variation search, a transposition table that survives between moves, MVV-LVA + killer +
history ordering, check extension, null move pruning, late move reductions, reverse
futility and futility pruning, and a capture/promotion quiescence with delta pruning.
Mates, stalemates, repetitions (the harness only sends a FEN, so the agent remembers every
position it has seen) and the fifty-move rule are scored by the rules, never by the network.

**Time.** About `time_left / 30 + 0.35 s` per move, capped at 6 s and at a quarter of the
clock, checked every 128 nodes. Depth 1 always runs unclocked, and a fallback move exists
before any search starts. Under 120 ms the fallback goes out untouched.

**Training (Giraffe's three stages, adapted).**

1. `training/gen_positions.py` — noisy self-play between one-ply handcrafted players gives
   240k diverse positions. `training/relabel.py` keeps the quiet ones (not in check, no
   capture that changes the value) and labels each with a depth-2 alpha-beta search of the
   handcrafted evaluator, storing the static score alongside so `label - static` is the
   residual target. No engine other than this one labels anything.
2. `training/bootstrap.py` — supervised pretraining of the torch twin on the residuals
   (MSE in tanh space), exported as the flat weight file.
3. `training/tdleaf.py` — TD-Leaf(lambda) self-play: fixed-depth searches with the current
   evaluator, principal-variation leaves trained towards discounted temporal differences of
   the search scores (minus the leaf's static score, so the network keeps fitting the
   residual), terminal values from the rules, a replay buffer, a slice of the supervised
   data as an anchor, and fixed-depth arena gating of checkpoints against the last accepted one.

`training/selfplay_arena.py` plays the identical search with two evaluators from paired
random openings, which is the controlled experiment the architecture brief asks for.

## Running

All commands use the project interpreter from this directory.

```
python -m unittest discover -s tests           # 30 tests: rules, clocks, symmetry, parity
python bench.py                                # latency / nodes/s / depth
python training/gen_positions.py --positions 240000 --workers 6
python training/relabel.py --data training/data/bootstrap.npz --depth 2 --workers 4
python training/bootstrap.py --data training/data/search_d2.npz --epochs 30 --out models/giraffe.npz
python training/tdleaf.py --init models/giraffe.npz --iterations 20 --gate-depth 3
python training/selfplay_arena.py --a models/giraffe.npz --b hce --pairs 20 --budget 1e9 --depth 3
GIRAFFE_EVAL=hce ...                           # run the same agent with the control evaluator
```

Harness games: from the repository root,
`python -m harness.arena --agent agents/30_giraffe --opponent baselines/minimax --games 10`.

## Notes for the judge

- The weights in `models/giraffe.npz` were produced by the scripts in `training/` from
  positions this project generated itself. No engine other than this one labelled them.
- numba compiles at import (cold: a few seconds on an idle core) and caches under the temp
  directory; if the cache location is unusable the import retries without a cache.
- `torch` is not imported at play time; only numpy, numba and python-chess are.
