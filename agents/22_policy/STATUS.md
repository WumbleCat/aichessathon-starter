# 22_policy — status checkpoint (for a fresh session)

Updated: 2026-09-05 03:50 (session 2)

## Where things stand

- Engine (`pn_search.py`, `pn_eval.py`), encoding, numpy inference, data generator, trainer: written
  and committed on branch `feature/agent-22-policy` (6 commits, not yet merged into main).
- `tests/test_agent.py` and `tests/test_encoding.py` pass; `tests/test_policy_inference.py` passes
  (numpy == torch to 5e-7; ONNX export fails on this machine with
  `onnx has no attribute load_model_from_string`, numpy backend is the default so it does not matter).
- Search speed: ~14k nodes/s CPU time at depth 4 (`tests/bench_search.py 4 0`) on the loaded box.
- Training data: 224 shards, 239,323 positions, 6.3M move labels in `training/data/`
  (git-ignored). Teacher depth histogram {1: 52k, 2: 132k, 3: 48k, 4: 7k}.
- Inference cost (CPU ms per prior call, numpy): C32B3 1.6, C48B4 2.3, C64B4 3.3, C64B5 3.9.

## Not done yet (in order)

1. TRAIN the network: `models/policy.npz` does not exist. Command:
   `python training/train.py --min-depth 2 --channels 64 --blocks 5 --epochs 6 --threads 4`
   (writes models/policy.npz + models/train_log.txt; the trainer exports on every val improvement,
   so a partially trained model is usable at any time).
2. Paired A/B arena: policy vs `variants/nopolicy` (`tests/paired_arena.py`), plus `variants/searchless`.
3. Write RESULTS.md (README refers to it), merge the branch into main.
4. Build the zip and check size (< 50 MB) and init time (< 90 s).

## Known issues

- Earlier arena results in `results/control_vs_15ext.json` were run on a machine with 100+
  concurrent processes: depth 1, 128 nodes per move, 10 flags. They say nothing about the engine.
  Re-run when the box is quieter, or compare variants pairwise at the same time.

## Session 2 log

- 04:17 lint: `ruff check agents/22_policy` and `ruff format` (pn_eval.py excluded so the PST
  tables keep their 8-per-row layout) are clean. `tests/check_submission.py` added (zip + size +
  import time + probe moves in a fresh interpreter).
- 04:17 training relaunched with `--threads 1`: on this saturated box torch with 4 threads took
  12 s per step (thread oversubscription), 1 thread takes 0.9 s. Log: `training/train_c48b4_t1.log`.
  Expected: ~10 min per epoch, 6 epochs, exports models/policy.npz on every val improvement.
- 04:31-05:20 training progress (val, 9,290 positions): ep0 top1 .324 top3 .571 | ep1 .349/.627 |
  ep2 .368/.637 | ep3 .371/.653. Offline `tests/eval_policy.py` on the ep0 model: top1 .334 vs
  hand-crafted ordering .218, cp lost vs teacher 127 vs 230.
- 05:00 `pn_search.Searcher` got a `clock` argument (default perf_counter). `tests/cpu_match.py`
  plays paired in-process matches on `time.process_time` so A/B results do not depend on the
  machine load (wall-clock arenas were meaningless: python startup 5 s, `import numpy` 29 s).
- Lint cleanup committed to `feature/agent-22-policy` as 852b44c (throwaway index, working tree
  untouched). Not yet committed: pn_search clock, tests/cpu_match.py, models/policy.npz, results.
