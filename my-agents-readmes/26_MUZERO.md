# MuZero for Chess

## Goal

Implement the MuZero research architecture:

```text
observation -> representation network
latent state + action -> dynamics network -> next latent state + reward
latent state -> prediction network -> policy + value
MCTS plans in latent space
```

This is included because it is a major RL methodology, **not because it is recommended for this contest**.

## AI Chessathon constraints you MUST design around

Target the current competition API:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    # return one legal move in UCI notation
```

Deployment constraints from the official docs:

- ZIP size: **50 MB maximum uncompressed**.
- `agent.py` must be at the ZIP root.
- One dedicated **CPU core**, **2 GB RAM**, **no GPU**, **no network**.
- Python 3.12.
- Preinstalled: `torch` (CPU), `numpy`, `python-chess`, `onnxruntime`, `numba`.
- 90 s initialization budget before the game clock starts; load weights at import time.
- Main clock: 120 s + 0.5 s/move.
- Native binaries in the ZIP are rejected.
- Published third-party chess engines and ports/translations are prohibited.
- A shipped neural network must be **trained by this team from scratch**.
- Training on positions labelled by an existing engine is explicitly allowed.
- Do **not** ship Stockfish/Lc0/Maia weights, fine-tuned versions of them, or a runtime lookup database of engine moves/evaluations.
- `.onnx`, `.safetensors`, and `.pt` weights are allowed if they are your own trained network.

Official rules/docs: https://aichessathon.com/docs

### Required engineering rule

Concepts and algorithms from papers are fair game. **Do not copy, port, translate, wrap, or mechanically reproduce third-party engine source.** Implement the described algorithm independently and be able to explain it.


## Core components

### Representation function `h`

```text
board observation -> latent state
```

### Dynamics function `g`

```text
latent state + action -> next latent state, predicted reward
```

### Prediction function `f`

```text
latent state -> policy, value
```

Search does not call the real chess transition after the root; it rolls the learned latent dynamics.

## Why this is inefficient for Chessathon

Chess already gives you a perfect, cheap simulator through `python-chess`.

AlphaZero can therefore do:

```text
real legal chess move -> exact next board
```

MuZero instead spends capacity learning an internal transition representation because it was designed to succeed even when dynamics are unknown.

For this contest, that is solving a problem you do not have.

## If implementing for research

### Inputs
Encode board state including:
- pieces;
- side;
- castling;
- repetition/clock information if relevant.

### Actions
Fixed move/action index with legal masking at the real root. In latent rollout, action validity becomes challenging because the model is not explicitly given the rules unless you deliberately combine it with the true simulator.

### Training sequence
Store trajectories and unroll K recurrent steps:

```text
s_t
 a_t
 r_90.884684739
 policy target
 value target
 ...
```

Train predictions at each unrolled step.

### Search
Use MCTS over latent states with policy priors and value estimates.

### Contest adaptation
If you allow real chess legality at every simulated step, you are moving back toward AlphaZero and losing the main reason to use MuZero. That is acceptable as an experiment but should be named accurately.

## Mandatory validation

Before comparing Elo, make these tests pass:

1. `get_move(fen, time_left_ms)` always returns a legal UCI move.
2. Test castling, promotion, en-passant, check evasion, mate and stalemate positions.
3. Test at very low clocks (`time_left_ms=50`, `200`, `1000`) and verify no flag/crash.
4. Test repeated calls in the same process; model/search state must remain valid.
5. Verify initialization is below 90 s.
6. Measure median and p99 move time.
7. Run at least hundreds of paired games against the supplied baselines.
8. Run both colours from the same starting positions.
9. Track W/D/L, Elo estimate, nodes/s, average depth and crashes.
10. Build the final ZIP and verify **uncompressed** size is < 50 MB.
11. Run in an offline environment with only contest-approved packages.
12. Save training scripts, dataset-generation scripts, seeds and provenance so the team can explain any shipped model.


## Critical MuZero tests

- representation of identical board must be deterministic in eval mode;
- dynamics unroll should not numerically explode;
- reward/value/policy perspective must be consistent;
- latent search must not return an illegal real root move;
- compare learned-dynamics predictions against actual next-position targets offline.

## Feasibility verdict

**Do not prioritize.** Use the exact game rules and spend compute on better search/evaluation instead.

## Papers

1. Julian Schrittwieser et al., **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**, arXiv: https://arxiv.org/abs/1911.08265
2. Nature version: https://doi.org/10.1038/s41586-020-03051-4
3. AlphaZero comparison: https://arxiv.org/abs/1712.01815
4. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Only implement MuZero after the main contest engine is complete. Keep it in a separate experimental branch and require it to beat an AlphaZero-style network of similar size before considering deployment.
