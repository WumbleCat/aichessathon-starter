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
- Tests: 26 of 26 pass on the final code (`tests/`, 2026-09-05 08:40 run); perft 4 kiwipete
  4,085,603 nodes, 8.6 Mnps when the box was quieter (1.1 Mnps in the final run).
- Compile in that final run: 152.6 s CPU under ~80 competing processes (119 s when quieter).
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
| h256 epoch 12 | row-view nnue.py | equal time 0.1 s | 40 | +23 =6 -11 | 65.0% | +108 | 114k (cpu) | 694k (cpu) |
| h256_long (30 ep) vs h256 epoch 12 | row-view nnue.py | 20,000 | 40 | +17 =5 -18 | 48.8% | -9 | 32.4k (wall) | 78.3k (wall) |

The 30-epoch net's 20 cp better validation error bought no playing strength (its train/val
gap was three times larger; the validation split was by position, so adjacent plies of the
~12k source games leak across it), so the epoch-12 net stays shipped and `train.py` now splits
validation by game. Two nets of identical shape also differed 2.4x in wall node rate, so the
per-node cost depends on how the search behaves under a given evaluation, not on arithmetic.

The node-rate gap did not move with the `evaluate`/`update` rewrite because it was never in
the engine: the A/B tool timed each side by wall clock on a swapping machine. Measured in
CPU time in one process (`prof_search.py`, 150k nodes from three positions each):

| searcher | eval | nodes | CPU s | knps |
|---|---|---|---|---|
| PSQT searcher | PSQT | 451,584 | 0.55 | 826 |
| NNUE searcher (H=256) | NNUE | 451,585 | 0.83 | 545 |
| NNUE searcher (H=256) | PSQT | 451,584 | 0.53 | 850 |

Replaying a 136-ply game through both searchers with the TT kept between moves (20k nodes
per move, `prof_game.py`): NNUE 2,586,861 nodes in 4.56-4.80 s CPU (539-567 knps), PSQT
2,544,234 nodes in 3.59 s (708 knps). So the network costs 1.3-1.5x per node, worth well
under a ply, against +190 Elo at equal nodes. The 6x CPU ratio printed by the equal-time A/B
is an artefact of Windows `process_time` ticking in 15.6 ms steps on 0.1 s moves.
`selfplay_ab.py` now reports node rates by CPU time as well as wall time.

`tools/vs_bot.py` against `my-agents/10_principal_variation_search` (the team's python PVS
bot, a 40-0 scorer against the four baselines) at 0.5 s per move, both sides, 10 games:
+9 =0 -1 (90%, about +380 Elo); the loss was a material adjudication at the 240-ply cap.
Under this load the python bot only reached depth 1-3, so the margin is inflated.

`harness.play` vs `baselines/greedy` at 120 s + 0.5 s: draw by threefold repetition, played
entirely by the python fallback (the compile outlasted the game on this box). The fallback now
treats root moves that repeat a seen position as draws.

## Submission

`harness.package` from `agents/21_nnue`: 6 files, 217,821 bytes zipped, 468,036 unzipped
(limit 50,000,000): `agent.py cboard.py csearch.py jitconf.py nnue.py weights/nnue.safetensors`.
