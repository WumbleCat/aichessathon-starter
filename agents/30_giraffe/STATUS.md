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

## Finding and redesign (this session)

The controlled experiment at fixed depth 3 (`results/selfplay_net_vs_hce_depth3.txt`):
the bootstrap net lost 4-20 (-280 Elo) to the handcrafted static score with the identical
search. Probe on 400 positions: mean |net - label| was 103 cp on quiet positions (p90
317 cp), 162 cp on tactical ones. A net that misjudges material by a pawn or more is worse
than counting it exactly.

Redesign, already in the code: `net_eval = hce static + network residual`
(`giraffe_eval.net_eval_bb`, `OUT_SCALE` 600). The residual is trained on quiet positions
towards `depth-2 search score - static` (`training/relabel.py` builds
`training/data/search_d2.npz` incrementally; `bootstrap.py` and `tdleaf.py` subtract the
`static` array). Tests pass. **The shipped `models/giraffe.npz` is still the OLD absolute
net and must be replaced by a residual net before anything ships** (until then the agent
plays hce + a bogus residual; still legal, but weak).

## Open items (in priority order)

1. Let `relabel.py` finish or reach ~50k positions (it saves every 10 chunks; the machine
   runs ~90 python processes, so a chunk of 1000 takes minutes). Log: `results/relabel_d2.log`.
2. `bootstrap.py --data training/data/search_d2.npz --out models/giraffe.npz --epochs 30`.
3. Controlled arena at fixed depth 3, net (static + residual) vs hce, 12+ pairs. Ship the
   net only if it scores >= 50%; otherwise ship with `GIRAFFE_EVAL=hce` semantics (make
   the default evaluator hce in agent.py) and say so in RESULTS.md.
4. TD-Leaf (`tdleaf.py --gate-depth 3`) if time allows.
5. Harness games vs `baselines/minimax` and `baselines/greedy`, RESULTS.md, commit on
   `feature/agent-30-giraffe` (throwaway-index workflow; other sessions share the tree),
   `--no-ff` merge into main.
