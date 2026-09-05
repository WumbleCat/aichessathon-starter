# 22_policy — results

All measurements below were taken on 2026-09-05 on a shared 16-core Windows box that was running
four other training jobs and several arenas at the same time (100% CPU load, 2 to 4 GB of free
memory, `import numpy` alone took up to 29 s). Where it matters, numbers are given in CPU time
(`time.process_time`) rather than wall time, and the paired match runs both sides on a CPU-time
clock (`tests/cpu_match.py`) so that the comparison does not depend on the load.

## Shipped model

| | |
|---|---|
| architecture | residual CNN, 48 channels, 4 blocks, 18x8x8 in, 4672 logits out (+ aux value head) |
| parameters | 215,166 |
| weights file | `models/policy.npz`, 0.87 MB (batch-norm folded) |
| inference | pure numpy, 2.3 ms CPU per `prior()` call (im2col matmuls, BLAS pinned to 1 thread) |
| training data | 239,323 self-play positions labelled by this project's own search; 187k kept (teacher depth >= 2), 5% of shards held out |
| targets | softmax(teacher score / 50 cp) over all legal moves; aux value = tanh(best score / 400) |
| training | AdamW, lr 2e-3 one-cycle, batch 256, 6 epochs, 81 min on one core (`training/train_c48b4_t1.log`) |
| provenance | `training/gen_data.py` seeds 1-6, `training/train.py --min-depth 2 --channels 48 --blocks 4 --epochs 6 --threads 1` |

Net sizes considered (CPU ms per prior call, numpy back end, `tests/bench_inference.py`):
C32B3 1.6 ms (89k params), **C48B4 2.3 ms (215k)**, C64B4 3.3 ms (365k), C64B5 3.9 ms (439k).
C48B4 was chosen because one call costs about 35 nodes of search (14k nodes/s) and the training
run fit in the time available on the loaded machine; the teacher is only depth 2-3, so a larger
net would mostly learn the teacher's noise.

## Offline policy metrics (`tests/eval_policy.py`, 9,290 held-out positions)

| phase | n | top-1 | top-3 | top-5 | soft CE | cp lost by top-1 | hand-crafted top-1 | hand-crafted cp lost |
|---|---|---|---|---|---|---|---|---|
| all | 9290 | 0.383 | 0.665 | 0.769 | 2.66 | 87 | 0.194 | 211 |
| opening | 3102 | 0.423 | 0.742 | 0.831 | 2.87 | 64 | 0.171 | 220 |
| middlegame | 3319 | 0.357 | 0.611 | 0.715 | 2.68 | 106 | 0.211 | 227 |
| endgame | 2869 | 0.369 | 0.645 | 0.766 | 2.42 | 89 | 0.198 | 183 |

"Hand-crafted" is the ordering the engine uses without a network at a fresh node: winning
captures and promotions by MVV-LVA, then quiet moves in generation order (no TT move, killers or
history, which only exist once a subtree has been searched). "cp lost" is the teacher score of
the best move minus the teacher score of the move ranked first.

Validation curve (`models/train_log.txt`): top-1 0.324 / 0.349 / 0.368 / 0.371 / 0.383 / 0.383
after epochs 1-6; the epoch-5 weights had the lowest validation loss and are the ones shipped.

## Searchless play (`tests/cpu_match.py --a searchless --b nopolicy`)

Playing the network's top move with no search at all lost 6/6 games (3 openings, both colours)
against the same engine searching 0.25 s CPU per move. The policy is a move-ordering aid, not an
engine; this confirms the README's expectation and is why searchless play is only a baseline.

## Policy for move ordering vs hand-crafted ordering

`tests/cpu_match.py --a policy --b nopolicy --pairs 8 --budget 0.25 --seed 11`
(`results/cpu_policy_vs_nopolicy.json`): 8 random openings, both colours, 0.25 s CPU per move
for each side, in one process.

| side | score | avg depth | avg nodes / move | prior calls / move | CPU s / move |
|---|---|---|---|---|---|
| policy (root + depth >= 4, LMR adjust) | 7 wins, 2 draws, 7 losses = **8.0 / 16** | 3.68 | 2,956 | 6.1 | 0.205 |
| nopolicy (TT / MVV-LVA / killers / history) | 8.0 / 16 | 3.85 | 3,201 | 0 | 0.200 |

Colour split: the policy side scored 6/8 as White and 2/8 as Black; 14 of 16 games ended in
checkmate. The 16-game sample is too small to separate the two (a 95% interval on 50% is about
+/- 25 percentage points), so the honest reading is: at this budget the network's ordering gain
and its inference cost (6 calls x 2.3 ms = 7% of the move budget, 8% fewer nodes) cancel out.
The budget matters: at 0.25 s CPU the search only reaches depth 3-4, so interior nodes with
remaining depth >= 4 are rare and the prior is used almost only at the root, where iterative
deepening already orders the previous best move first. At the contest control (120 s + 0.5 s,
roughly 4 s per move, depth 5-6) the prior fires at many more nodes; that is the setting to
measure next and it could not be run on this machine in the time available (a 16-game match at
1 s CPU per move takes about four hours here).

What the offline numbers say the search should gain: the network puts the teacher's best move
first in 38% of positions against 19% for the hand-crafted first move, and its first choice
loses 87 cp instead of 211 cp, so the first quiet move tried at a policy node is much more
often the cut move. The engine is shipped with the policy on (`PN_USE_POLICY=1`,
`PN_POLICY_MIN_DEPTH=4`) because it is not worse at short budgets and has more room to help at
long ones; `variants/nopolicy` is the control if that ever needs re-checking.

## Engine health (`tests/test_agent.py`, `tests/check_submission.py`)

* Every special-move case returns a legal move (castling both sides, en passant, promotions
  including underpromotion positions, check evasion, mate in one found, stalemate avoided).
* Clock: replies in 0-1 ms at 50 and 100 ms left, 28 ms at 1 s, 0.8 s at 30 s, 3.6 s at 120 s.
* A 45-ply game against itself in one process with repetition memory kept between calls.
* Submission zip: 0.92 MB unzipped (agent.py, four `pn_*.py` modules, `models/policy.npz`,
  `models/train_log.txt`), import 1.7-1.9 s in a fresh interpreter on the loaded machine,
  network loaded, worst probe move 0.08 s with 2 s on the clock.
* Search speed without the network: ~14k nodes/s CPU at depth 4 (`tests/bench_search.py 4 0`).

## Earlier wall-clock arenas (previous session, `results/`)

`arena_control_*`: the engine without a network beat `baselines/greedy` 10-0 and
`baselines/minimax` 10-0 at 10 s + 0.1 s. `control_vs_15ext.json` (3-6-11 against
`my-agents/15_selective_extensions`) was played while 100+ processes shared the machine: the
engine reached depth 1 with 128 nodes per move and flagged 10 times, so it says nothing about
strength. Re-run `tests/paired_arena.py` on a quiet machine before quoting it.

## What to try next

* Iterate the teacher: label with the policy-ordered engine at depth 4-5 and retrain (the data
  generator only needs `Searcher(prior=...)`).
* Benchmark `policy_min_depth` 3 and 5 and root-only with `tests/cpu_match.py` (`--a d3`,
  `--a d5`, `--a rootonly`).
* Use the auxiliary value head as a cheap tie-break in quiescence stand-pat.
