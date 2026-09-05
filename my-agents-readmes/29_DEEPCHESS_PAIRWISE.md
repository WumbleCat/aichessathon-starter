# DeepChess-Style Pairwise Position Evaluation

## Goal

Reproduce the conceptual method of DeepChess: learn a chess representation and train a network to decide which of two positions is preferable, then use that learned comparator/evaluation inside a chess-playing system.

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


## Paper concept

DeepChess used:

1. unsupervised feature learning from chess positions;
2. supervised training to compare two chess positions;
3. learned representations rather than manually coded chess features.

The important design idea is **pairwise preference learning**.

## Contest-oriented adaptation

A direct pairwise comparator is awkward inside alpha-beta because search wants a scalar value.

Two options:

### A. Faithful comparator

Train:

```text
(position A, position B) -> probability A better than B
```

At a root, compare child positions in a tournament/bracket.

Problem: alpha-beta bounds become difficult because comparator scores are not naturally transitive scalar utilities.

### B. Recommended adaptation

Train a shared position encoder:

```text
position -> latent -> scalar value
```

Generate pairwise loss:

```text
sigmoid(V(A) - V(B))
```

This preserves DeepChess's preference objective while producing a scalar compatible with alpha-beta.

## Data

Possible label construction:

- positions from winning vs losing game trajectories;
- teacher evaluations;
- pairs where teacher score difference exceeds a margin.

Balance color and game phase.

## Model

Keep it CPU-small:

```text
board feature vector
-> dense layer
-> dense layer
-> latent representation
-> scalar value
```

or a tiny board CNN.

Do not reproduce published weights.

## Pairwise loss

For target `y=1` when A preferred:

```text
logit = V(A) - V(B)
loss = binary_cross_entropy_with_logits(logit, y)
```

Add direct value regression as a multi-task target if teacher scores are available.

## Integration

The scalar variant can replace handcrafted eval.

Test:

```text
handcrafted
DeepChess-style scalar
blend
```

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


## Debugging

- If A==B, predicted preference should be near 0.5.
- Swapping A/B should invert probability.
- Color normalization must not reverse labels accidentally.
- Rank a set of positions by teacher score and calculate Spearman correlation.

## Feasibility verdict

Useful research route but NNUE is much better aligned with very high-frequency CPU evaluation.

## Paper

Eli David, Nathan S. Netanyahu, Lior Wolf, **DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess**: https://arxiv.org/abs/1711.09667

Also compare:
- Giraffe: https://arxiv.org/abs/1509.01549
- NNUE design: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- contest rules: https://aichessathon.com/docs

## Coding-agent instruction

Implement pairwise learning cleanly, but expose a scalar value head if the final engine uses alpha-beta. Benchmark against the same-sized NNUE before deployment.
