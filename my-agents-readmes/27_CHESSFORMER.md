# Chessformer

## Goal

Implement a **small Chessformer-inspired model from scratch**, based on the 2026 ICLR paper, and test it as:

1. a policy/value network for search; or
2. a supervised move/value model.

Do not ship published Chessformer/Lc0 weights.

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


## Paper idea

Chessformer changes the representation and attention mechanism to respect chess geometry.

Major ideas described by the paper:

- process the **64 board squares as tokens**;
- use a dynamic/geometric positional mechanism;
- use **Geometric Attention Bias (GAB)** to encode chess-relevant square relationships;
- use an attention-based policy output;
- use the architecture across engine-strength and human-move modeling tasks.

## Contest adaptation

The original high-strength use is not designed around a single CPU core and 50 MB model cap. Build a tiny version.

Example search space:

```text
embedding dimension: 64 / 96 / 128
layers: 2 / 4 / 6
attention heads: 4 / 8
MLP ratio: 2-4
```

Measure serialized size before long training.

## Token features

For each square encode:

```text
piece identity / empty
piece color
square/file/rank
side to move (global or added to every token)
castling rights
optional en-passant relation
```

Keep a consistent orientation, preferably normalize side-to-move perspective.

## Geometric Attention Bias

Do not simply use a vanilla Transformer and call it Chessformer.

Implement a learnable bias based on geometric relationships between source and target squares, e.g.:

```text
same rank
same file
same diagonal
relative dx/dy
knight-like relation
distance / direction buckets
```

Follow the paper's mechanism as closely as practical while writing your own implementation.

## Outputs

### Policy
Map contextualized square embeddings to move logits. The paper uses an attention-based policy design; for a contest prototype, ensure all legal chess moves including promotions can be represented.

### Value
Add pooled/global representation -> MLP -> scalar/WDL.

## Training options

### Supervised engine distillation
Recommended:
- Stockfish-labelled positions offline;
- teacher move/action values;
- train policy and/or value.

### Self-play
Possible but much more expensive.

## Deployment

Potential uses ranked for this contest:

```text
1. root-only policy ordering
2. shallow-node policy/value guidance
3. low-simulation MCTS
4. searchless play
```

Avoid invoking a Transformer at every alpha-beta leaf unless benchmarks prove it is cheap enough.

Try ONNX CPU export and compare with PyTorch CPU.

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


## Chessformer-specific tests

- exactly 64 square tokens;
- geometric-bias tensor indices correct under board flips;
- promotion moves map uniquely;
- legal-mask output;
- compare model prediction before/after color normalization;
- measure inference latency at batch 1, because contest search mostly cares about latency rather than throughput.

## Feasibility verdict

Scientifically cutting-edge, but **lower contest priority than NNUE** due to CPU latency and model-size pressure.

## Papers / references

1. Daniel Monroe et al., **Chessformer: A Unified Architecture for Chess Modeling**, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d167db04a90885ad5208fe8b273668b-Abstract-Conference.html
2. Earlier Chessformer work, **Mastering Chess with a Transformer Model**: https://arxiv.org/abs/2409.12272
3. AlphaZero policy/value search context: https://arxiv.org/abs/1712.01815
4. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Implement a deliberately small Chessformer-inspired model from random initialization. Benchmark batch-1 CPU latency before committing to large training. Never download/ship a published Chessformer or Lc0 network.
