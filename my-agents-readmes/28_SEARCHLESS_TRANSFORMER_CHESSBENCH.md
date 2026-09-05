# Searchless Transformer / ChessBench-Style Distillation

## Goal

Train a Transformer to approximate the results of expensive engine planning **without performing explicit search at tournament time**.

This follows the research direction of *Amortized Planning with Large-Scale Transformers: A Case Study on Chess*.

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


## Paper result

The ChessBench work uses:

- 10 million chess games;
- about 15 billion labelled data points;
- Stockfish 16 legal-move/value annotations;
- supervised Transformers up to 270M parameters;
- targets including state value, action value and behavioral cloning.

The largest model achieved reported Lichess blitz strength around 2895 without explicit search.

Do not attempt to ship their weights. Your model must be trained by you.

## Contest challenge

270M FP32 parameters alone are roughly >1 GB, far above this contest's 50 MB uncompressed ZIP limit.

Therefore build a **distilled small variant**, e.g.:

```text
1M-10M parameters
```

and/or quantize your own model after training.

Remember: model size and CPU latency matter simultaneously.

## Recommended target: action values

Instead of directly predicting a move label, predict a score for candidate actions:

```text
Q(s, a)
```

At inference:

1. generate legal moves using `python-chess`;
2. score each legal move/model action;
3. choose max score.

This follows the paper's finding that action-value prediction can encode nontrivial planning information.

## Dataset generation

Offline:

1. sample millions of diverse positions;
2. ask teacher engine for legal moves;
3. obtain a score/value for each chosen candidate move;
4. store compact records.

Because scoring *every* legal action is expensive, alternatives:

- teacher MultiPV;
- top K + random legal negatives;
- shallow child evaluation;
- deeper labels for hard positions.

Do not ship this teacher database in the final ZIP; it is training data only.

## Architecture options

A small Transformer can use:

### Board-token representation
One token per square plus global state.

### FEN-like/sequential representation
Closer to some language-model implementations but less structurally natural.

For a contest implementation, 64 square tokens are easier to legality-check and debug.

## Training losses

Action-value regression:

```text
Huber/MSE(predicted_Q, teacher_Q)
```

Ranking loss:

```text
ensure best teacher move > weaker alternatives
```

Policy distribution:

```text
softmax teacher scores -> cross entropy/KL
```

Test combinations.

## Inference optimization

- legal-mask first;
- avoid generating logits for unnecessary moves if architecture supports candidate scoring;
- export to ONNX;
- test INT8 quantization;
- load once at import time;
- use one inference per move if searchless.

## Hybrid extension

Even if searchless performance is mediocre, the model may be valuable as:

```text
root move ordering
or
candidate pruning before classical search
```

Be conservative with candidate pruning: never permanently drop forcing moves until proven safe by games.

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


## Metrics

In addition to Elo:

- top-1 teacher move match;
- average centipawn loss;
- tactical puzzle accuracy;
- endgame accuracy;
- inference ms/position;
- model MB.

## Paper / code references

1. Anian Ruoss et al., **Amortized Planning with Large-Scale Transformers: A Case Study on Chess** (NeurIPS 2024): https://arxiv.org/abs/2402.04494
2. NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/78f0db30c39c850de728c769f42fc903-Abstract-Conference.html
3. DeepMind research implementation: https://github.com/google-deepmind/searchless_chess
4. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Recreate the *methodological idea*, not the published model. Train a much smaller model from scratch on your own teacher-labelled data, and aggressively benchmark size/latency. Expect classical search to remain stronger under this hardware constraint.
