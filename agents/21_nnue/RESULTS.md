# 21_nnue — measurements

All numbers from 2026-09-05 on a 16-core Windows box shared with 60-90 other Python
processes, so wall-clock figures are inflated 3-30x; CPU seconds and same-process
comparisons are the reliable ones.

## Training (`training/train.py`, 865,401 positions, Stockfish 17.1 at 6000 nodes as teacher)

| run | args | epochs | val MSE (sigmoid space) | val MAE cp | val sign acc |
|---|---|---|---|---|---|
| h256 | `--hidden 256 --epochs 12 --threads 4 --lambda-result 0.1 --seed 1` | 12 | 0.00494 | 134.3 | 95.0% |
| h256_long | `--hidden 256 --epochs 30 --threads 1 --lambda-result 0.0 --seed 2` | 30 | 0.00318 | 113.8 | 96.0% |

Single-thread training was 3x faster per epoch than 4 threads on the loaded box.
`val_loss` is against the blended target, so it is only comparable within one `--lambda-result`.

Quantised export (`training/export.py`, int16, QA 255 / QB 64 / SCALE 400): 395,588 bytes,
worst-case accumulator 1931 of 32767. Per-pawn probe of the h256 net from the start position
(side-to-move cp, start = +31): removing a black pawn on a7/b7/c7/d7/e7/f7/g7/h7 gives
+0/+91/+61/+54/+48/+151/+123/+9. Material: queen ~1000, rook ~560, KQ v K +601.
The h256_long net (models/nnue_h256_long.safetensors, |W1| max 121, worst-case accumulator
2617): same probe -6/+117/+98/+75/+77/+167/+152/+39; in a quiet middlegame its black pawns
are worth +40 to +158 each versus -9 to +140 for h256. Queen 997, KQ v K 553, KR v K 413.

## Engine

- Compile of the numba search: 119-125 s CPU (fresh process, no cache). `agent.py` builds it
  in a thread, waits up to 70 s at import, and plays a python-chess alpha-beta meanwhile.
- Tests: 25 of 25 pass (`tests/`); perft 4 kiwipete 4,085,603 nodes, 8.6 Mnps.
- Clock levels (time_left -> used): 50 ms 2.9 ms, 100 ms 8.0 ms, 1 s 34 ms, 5 s 334 ms,
  30 s 1.24 s, 120 s 4.24 s.
- Micro-benchmark after the row-view rewrite of `nnue.py` (same process, low contention):
  `evaluate` 0.15 us, `update` 0.10 us, `refresh` 1.17 us, make+unmake 0.08 us per move.
  Before the rewrite, under heavy contention: 8.7 / 1.85 / 38.7 / 2.4 us.

## Strength

`tools/selfplay_ab.py`: NNUE evaluation vs the PSQT fallback, identical search, node-limited,
both colours per random 8-ply opening.

| net | code | nodes/move | games | result (NNUE) | score | Elo | nps NNUE | nps PSQT |
|---|---|---|---|---|---|---|---|---|
| h256 epoch 4 | old nnue.py | 20,000 | 40 | +28 =3 -9 | 73.8% | +179 | 46.8k | 724k |
| h256 epoch 12 | row-view nnue.py | 20,000 | 40 | +28 =4 -8 | 75.0% | +191 | 48.7k | 732k |

The node-rate gap did not move with the `evaluate`/`update` rewrite because it was never in
the engine: the A/B tool timed each side by wall clock on a swapping machine. Measured in
CPU time in one process (`prof_search.py`, 150k nodes from three positions each):

| searcher | eval | nodes | CPU s | knps |
|---|---|---|---|---|
| PSQT searcher | PSQT | 451,584 | 0.55 | 826 |
| NNUE searcher (H=256) | NNUE | 451,585 | 0.83 | 545 |
| NNUE searcher (H=256) | PSQT | 451,584 | 0.53 | 850 |

So the network costs 1.5x per node, worth well under a ply, against +190 Elo at equal nodes.
`selfplay_ab.py` now reports node rates by CPU time as well as wall time.

`harness.play` vs `baselines/greedy` at 120 s + 0.5 s: draw by threefold repetition, played
entirely by the python fallback (the compile outlasted the game on this box). The fallback now
treats root moves that repeat a seen position as draws.

## Submission

`harness.package` from `agents/21_nnue`: 6 files, 217,540 bytes zipped, 467,703 unzipped
(limit 50,000,000): `agent.py cboard.py csearch.py jitconf.py nnue.py weights/nnue.safetensors`.
