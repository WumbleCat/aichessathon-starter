# Agent 30 (Giraffe) — working state

Updated 2026-09-05. The git branch `feature/agent-30-giraffe` is the checkpoint; the
filesystem under `agents/30_giraffe` is the source of truth if the branch and disk
disagree. Other Claude sessions share the working tree, so commits on this branch are
built with a throwaway index (`GIT_INDEX_FILE`), never by checking the branch out.

## Stage: complete but improvable

- Engine, features, residual network, training pipeline, tests, benchmarks and
  RESULTS.md are done. The shipped `models/giraffe.npz` is a residual net (static
  handcrafted score + learned correction) trained on 171,890 quiet positions labelled by
  a depth-2 search of the handcrafted evaluator.
- Controlled arena at fixed depth 3, same search: static + residual beats static alone
  12-3-5 (67.5%). The earlier absolute-value net lost 4-20. See RESULTS.md.
- 30/30 unit tests pass. Import ~20 s cold under load, <1 s with a warm numba cache.

## Lessons for whoever continues

- Never train a small MLP to replace the static score; train it as a residual over an
  exact material/PST score. That single change moved the net from -280 to +127 Elo.
- Torch with more than one thread spin-waits on this shared machine (an hour with no
  progress); `--threads 1` trains 172k samples in 6 s per epoch.
- Wall-clock arenas are noise here; use `selfplay_arena.py --budget 1e9 --depth 3`.
- `relabel.py` saves every 10 chunks, so partial datasets are usable while it runs.

## Open items, in priority order

1. Harness games with the real clock against `baselines/minimax` and `baselines/greedy`
   when the machine is quieter (results/arena_net_vs_minimax.txt has the loaded-machine run).
2. TD-Leaf refinement: `training/tdleaf.py --init models/giraffe.npz --iterations 20
   --gate-depth 3 --workers 3`. It works end to end (smoke-tested) but has not been run
   long enough to gate a checkpoint. Accept only if the fixed-depth gate says >= 50%.
3. Depth-3 labels (`relabel.py --depth 3`) would give the residual more to learn, at
   roughly 5x the labelling cost.
4. Search speed: the Python search loop dominates; packing the bitboards into one uint64
   array per call (what an earlier session started) trims call overhead only marginally.
5. Done: `feature/agent-30-giraffe` was merged into main (`--no-ff` equivalent, merge
   commit a94c672) and deleted. New work goes on a fresh `feature/agent-30-<topic>` branch.
