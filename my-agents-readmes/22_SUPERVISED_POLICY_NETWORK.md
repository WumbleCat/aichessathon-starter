# Supervised Policy Network

## Goal

Train a network from scratch to rank legal moves. Use it either as a searchless baseline or, preferably, to improve move ordering near the root of alpha-beta.

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


## Training data

Two legal routes:

### A. Engine teacher

Offline for each position:

```text
position -> teacher best move
```

Better:

```text
position -> teacher scores for many legal moves
```

Then train a soft target distribution instead of only one label.

Example soft target:

```text
p(a|s) proportional to exp(score(a)/temperature)
```

### B. Human games

Use PGN positions and the played move. This learns human move likelihood, not necessarily strongest move quality.

For winning a bot tournament, engine targets are generally more aligned.

## Action encoding

Do not let the model output arbitrary UCI strings.

Use a fixed action space. Two practical choices:

1. `from_square x to_square x promotion_type`, then mask illegal actions.
2. AlphaZero-like spatial move planes.

The simplest contest implementation is a fixed integer index for every potential UCI move class plus a legal mask.

## Input encoding

Small dense/CNN option:

```text
12 piece planes
side to move
castling rights
en-passant information
optional repetition/halfmove features
```

Keep architecture small enough for CPU.

## Loss

For one-hot teacher move:

```text
cross_entropy(logits, target_move)
```

For teacher distribution:

```text
cross_entropy / KL(target_distribution, predicted_distribution)
```

Always mask illegal moves at inference.

## Best use: move ordering

A policy network does not prove tactics. Use it selectively:

```text
at root / shallow nodes:
    policy(position)
    order legal moves by prior

deeper nodes:
    cheap TT/history/capture ordering
```

Why? One neural inference at every alpha-beta node can reduce depth more than the better ordering helps.

Benchmark policy usage at:

```text
root only
depth >= 6
PV nodes only
first N plies
never
```

## Searchless version

For a pure policy agent:

1. compute logits once;
2. mask illegal moves;
3. choose highest-probability legal move.

This is a useful latency baseline but is tactically fragile.

## Data curriculum

Include:

- quiet positions where several moves are similar;
- tactical positions;
- endgames;
- underpromotions;
- unusual starting positions, because the contest uses curated near-level positions rather than always starting from move 1.

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


## Policy-specific metrics

Offline:
- top-1 legal move accuracy;
- top-3/top-5 accuracy;
- cross entropy;
- teacher score loss of chosen move;
- accuracy by game phase.

Online:
- Elo when used searchless;
- Elo gain/loss when used only for move ordering;
- nodes/s reduction from network calls.

## References

1. AlphaZero: policy/value network and search policy target: https://arxiv.org/abs/1712.01815
2. Anian Ruoss et al., **Amortized Planning with Large-Scale Transformers: A Case Study on Chess** / ChessBench, including behavioral cloning and action-value prediction: https://arxiv.org/abs/2402.04494
3. DeepChess for an earlier learned chess representation/evaluation approach: https://arxiv.org/abs/1711.09667
4. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Implement the smallest policy network that can measurably improve root/shallow move ordering. Treat searchless policy play as a baseline, not the presumed final engine.
