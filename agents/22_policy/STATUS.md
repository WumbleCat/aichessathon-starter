# 22_policy — status checkpoint (for a fresh session)

Updated: 2026-09-05 07:15 (session 2)

## Where things stand

- Everything is implemented and tested; the trained network ships in `models/policy.npz`
  (48x4 residual CNN, 215k params, held-out top-1 0.383). See `RESULTS.md` for the numbers.
- Branch `feature/agent-22-policy` holds the work (commits 852b44c, 748187a and later); it is
  built with a throwaway git index so the shared working tree is never checked out. Merge into
  main with `git merge --no-ff feature/agent-22-policy` once the last results are committed.
- `tests/check_submission.py` passes: zip 0.92 MB unzipped, import < 2 s, network loaded.
- Torch checkpoints live in `training/checkpoints/` (git-ignored, outside the zip).

## Open items

1. `results/cpu_policy_vs_nopolicy_b1.json` (if present): a 4-pair match at 1.0 s CPU per move
   launched at the end of session 2 (`results/cpu_policy_vs_nopolicy_b1.log`). Add its summary to
   RESULTS.md next to the 0.25 s result (8.0/16, a dead heat).
2. Optional: `tests/cpu_match.py --a d3|d5|rootonly --b nopolicy` to tune `PN_POLICY_MIN_DEPTH`.
3. Optional: re-run `tests/paired_arena.py` against baselines on a quiet machine (wall clock).
4. Next model: label positions with the policy-ordered engine at depth 4-5 and retrain.

## Known issues

- ONNX export fails on this machine (`onnx has no attribute load_model_from_string`); the numpy
  back end is the default and was verified equal to torch to 5e-7, so nothing depends on it.
- Old wall-clock arena results in `results/control_vs_15ext.json` were taken under extreme load
  (depth 1, 10 flags) and are not meaningful.

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
