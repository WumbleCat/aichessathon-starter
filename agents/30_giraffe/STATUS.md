# Agent 30 (Giraffe) — working state

Updated 2026-09-05 by the session that repaired the feature extractor. The git branch
`feature/agent-30-giraffe` is the checkpoint; the filesystem under `agents/30_giraffe` is
the source of truth if the branch and disk disagree.

## Stage

Core engine, bootstrap training and tests are complete. Stage 3 (TD-Leaf self-play
refinement) was started by an earlier session but never produced an accepted checkpoint;
`models/giraffe.npz` is still identical to `models/giraffe_boot.npz` (supervised on
240k self-generated positions labelled by the quiescence-resolved handcrafted eval).

## Done this session

- `giraffe_eval.py` was left mid-refactor (a packed-bitboard `features(bb, x)` signature
  with a body that still used `ep_square`); numba could not compile it and the agent did
  not import. Restored the twelve-argument bitboard signature. 30/30 unit tests pass.
- `bench.py` at 1.5 s/move on the loaded shared machine: net 35 us/eval, ~9.5k nodes/s,
  mean depth 4.2; hce 4.7 us/eval, ~7.2k nodes/s, mean depth 3.8.

## Open items (in priority order)

1. Controlled experiment net vs hce with the identical search, fixed depth
   (`training/selfplay_arena.py --budget 1e9 --depth 3`). Result file:
   `results/selfplay_net_vs_hce_depth3.txt`.
2. TD-Leaf refinement (`training/tdleaf.py`) with arena gating; ship only if it beats the
   bootstrap net in the controlled arena. Keep workers low, the machine is shared.
3. Harness games against `baselines/minimax` and `baselines/greedy` (earlier runs with the
   hce evaluator lost games to `init` and `flag` because the machine was saturated, not
   because of the agent; re-run when load allows and record in RESULTS.md).
4. Write RESULTS.md, then merge the branch into main with `--no-ff`.

## Ideas not yet tried

- Pack the bitboards into one uint64 array per call (what the earlier session was doing)
  to trim Python-to-numba call overhead; the evaluator is only ~30% of search time, so
  the python-chess search itself is the bigger cost.
- Increase `H_P` or add pawn-structure features; the bootstrap validation error (36 cp)
  is dominated by tactical positions where the label came from quiescence.
