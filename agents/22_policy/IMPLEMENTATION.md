# 22_policy — implementation plan

Architecture: **supervised policy network** used for move ordering inside an alpha-beta search
(README: `my-agents-readmes/22_SUPERVISED_POLICY_NETWORK.md`). Searchless policy play is kept
only as a latency/strength baseline.

## How the harness calls us

* `harness/runner.py` does `import agent` from the submission directory (first on `sys.path`) and
  then, per move, calls `agent.get_move(fen, time_left_ms)`; stdout is redirected to stderr.
* `harness/referee.py` measures wall time around that call, subtracts it from our clock, adds
  the increment after the move. A reply after `time_left_ms + 500 ms` grace is a flag; a bad
  move is a loss. The referee claims threefold repetition automatically.
* `harness/arena.py` alternates colours game by game (even games = we are white), default
  fast control 10 s + 0.1 s; `harness/play.py` plays one 120 s + 0.5 s game and accepts `--fen`.
* Only a bare FEN is sent, so repetition history must be kept by the agent itself.

## Constraints designed around

1 core, 2 GB, no GPU, no network, Python 3.12, `torch.set_num_threads(1)`, 90 s import budget,
50 MB unzipped. Only torch/numpy/python-chess/onnxruntime/numba are available.

## Components (all inside `agents/22_policy/`)

| file | role |
|---|---|
| `agent.py` | entry point; time management, repetition memory, fallback move, calls the searcher |
| `search.py` | iterative-deepening PVS + quiescence + TT + killers/history + NMP + LMR + futility |
| `evaluate.py` | tapered material + piece-square evaluation (fast, python-chess bitboards) |
| `encoding.py` | board -> 18x8x8 input planes; move <-> 4672-way action index (AlphaZero style 73 planes) |
| `policy.py` | network definition (small residual CNN) + numpy/onnx batch-1 inference wrapper |
| `models/policy.npz` | trained weights (own training, from random init) |
| `training/gen_data.py` | teacher self-play with noise; labels = teacher root scores per legal move |
| `training/train.py` | supervised training; soft targets `softmax(score / T)`; exports weights |
| `tests/` | legality, special moves, clock, repeated calls, encoding round-trip |

## Teacher

No external engine binary is used. The teacher is this project's own alpha-beta engine
(`search.py`) run at a fixed depth with an exact score for **every** legal root move. Positions
come from self-play games where moves are sampled from the teacher distribution with noise, from
random opening plies, so quiet, tactical and endgame positions are all present. Soft targets:
`p(a|s) ∝ exp(score(a)/T)`.

## Policy usage in search (to be benchmarked)

* root only
* root + nodes with `depth >= N` (N in 3..5)
* never (pure hand-crafted ordering) as the control
* policy prior also shrinks LMR reductions for high-prior moves

## Phases

1. Minimal legal agent (fallback move) — done first, keep as the safety net.
2. Full search engine, no network. Benchmark vs baselines.
3. Encoding + tests. Dataset generation. Training. Export.
4. Integrate the policy for ordering. A/B at fast time control, paired colours.
5. RESULTS.md with W/D/L, node rates, depths, inference latency, model size.
