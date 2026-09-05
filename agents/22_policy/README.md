# 22_policy — supervised policy network for move ordering

An alpha-beta engine whose move ordering at the root and at shallow interior nodes comes from a
small convolutional policy network trained from random initialisation by this project.
Searchless "play the top policy move" is kept only as a baseline.

Entry point: `agent.py` — `get_move(fen, time_left_ms) -> uci`.

## Files

| file | purpose |
|---|---|
| `agent.py` | entry point, clock budgeting, repetition memory, legal fallback, engine call |
| `pn_search.py` | iterative deepening, aspiration windows, PVS, TT, quiescence, killers/history, null move, LMR, futility, check extension; optional policy prior hook |
| `pn_eval.py` | hand-written tapered evaluation (material, piece-square tables, pawn structure, bishop pair) with a pawn-structure cache |
| `pn_encoding.py` | board -> 18x8x8 planes (side-to-move frame); move <-> 4672-way action index (AlphaZero-style 73 planes) |
| `pn_policy.py` | batch-1 inference: pure numpy (default, fastest here) or onnxruntime; `prior(board)` = softmax over legal moves |
| `models/policy.npz` | trained weights (folded batch-norm), produced by `training/train.py` |
| `training/gen_data.py` | teacher self-play labelling (this project's own search scores every legal move) |
| `training/train.py` | supervised training on soft targets, exports `models/policy.npz` |
| `training/model.py` | PyTorch network definition and export |
| `tests/` | legality/special-move/clock tests, encoding tests, inference equivalence, benchmarks, paired arena |
| `variants/` | one-line agent directories that run the same engine under different policy settings for A/B arenas |

All modules use a `pn_` prefix so nothing in the zip can shadow a standard library or
python-chess module (the zip goes first on `sys.path`).

## How the policy is used

`Searcher(prior=...)` asks the network for a prior at the root (always) and at any interior node
whose remaining depth is at least `policy_min_depth` (default 4, environment
`PN_POLICY_MIN_DEPTH`). Results are cached per position so iterative deepening never pays twice.
At those nodes:

* TT move first, then winning captures (MVV-LVA), promotions;
* quiet moves ordered by the network prior (killers and history only break ties);
* late move reductions are one ply smaller for moves with prior >= 0.20 and one ply larger for
  moves with prior < 0.02.

Elsewhere the ordering is the usual TT / captures / killers / history.

Environment switches (for experiments; defaults are the shipped configuration):
`PN_USE_POLICY`, `PN_POLICY_ROOT`, `PN_POLICY_MIN_DEPTH`, `PN_POLICY_LMR`, `PN_SEARCHLESS`,
`PN_BACKEND` (`numpy` / `onnx` / `auto`), `PN_MODEL_PATH`.

## Network

Input 18 planes (own/opponent P N B R Q K, four castling flags, en passant, ones).
Residual CNN: 3x3 stem, N residual blocks of two 3x3 convolutions with batch-norm, a policy head
(3x3 conv, 1x1 conv to 73 planes -> 4672 logits) and an auxiliary value head (used only during
training). Sizes trained are listed in `RESULTS.md`. Batch-norm is folded into the convolutions
at export; inference is an im2col matmul per convolution in numpy with BLAS pinned to one
thread.

## Training data and provenance

No external engine binary is used anywhere. `training/gen_data.py` plays noisy self-play with
this engine as the teacher: at every position an iterative deepening search scores **every**
legal root move (moves more than 300 cp below the best only get a bound), the target
distribution is `softmax(score / T)` with `T = 50 cp`, and the next move is sampled from that
distribution with a temperature (80 cp in the first 16 plies, 40 cp later) plus 2% uniform
random moves. Games start from the initial position or after 2 to 12 random plies, are
adjudicated when the teacher score exceeds 1000 cp for 4 consecutive plies, and are capped at
240 plies. Seeds, depth/node caps and shard formats are documented at the top of the script;
the training log is in `models/train_log.txt`.

## Reproduce

```
python training/gen_data.py --workers 8 --positions 300000 --seed 3 --depth 3 --nodes 6000
python training/train.py --channels 64 --blocks 5 --epochs 8
python tests/test_agent.py
python tests/paired_arena.py --agent agents/22_policy --opponent agents/22_policy/variants/nopolicy --pairs 20
```

## Contest checklist

* 1 core: BLAS thread environment variables are pinned to 1 before numpy is imported; no torch
  import at game time.
* 90 s init: importing the agent loads a ~2 MB `.npz`; well under a second.
* 50 MB: the whole directory including weights is a few MB.
* No network, no subprocesses, no writes.
* A legal fallback move is computed before the search starts and used if anything fails.
