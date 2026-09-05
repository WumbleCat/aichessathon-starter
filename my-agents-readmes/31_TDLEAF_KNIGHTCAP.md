# TDLeaf(lambda) / KnightCap-Style Learning

## Goal

Learn an evaluation function from game-tree-search outcomes using temporal-difference learning, following the TDLeaf(lambda) / KnightCap line of work.

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


## Core idea

Ordinary TD learning updates predictions from sequential states.

With chess search, the move decision depends on the **leaf selected by minimax/alpha-beta**, not merely the raw current board.

TDLeaf applies temporal-difference learning to the sequence of principal/selected leaf evaluations produced by game-tree search.

Conceptually:

```text
position s_t
 -> search
 -> selected leaf l_t
 -> evaluator V(l_t)
```

Then learn so consecutive selected leaf values and final result become consistent.

## Why use it

This is a path to automatically tune or learn evaluation without requiring a giant teacher-labelled dataset.

It is much cheaper than AlphaZero-style MCTS training if the evaluation model is small.

## Implementation path

### 1. Start with a differentiable evaluator

Options:
- linear handcrafted features;
- small MLP;
- tiny NNUE-like model.

### 2. Search

Use tested alpha-beta/PVS. During training, retain the leaf/feature state corresponding to each root decision.

### 3. Eligibility trace / lambda return

Implement TD(lambda) carefully from the paper.

For a sequence of predictions:

```text
V_0, V_1, ..., V_T
```

combine n-step temporal differences using lambda-weighted traces.

At game end, terminal reward anchors the trajectory.

### 4. Training mode

Use:
- self-play;
- online games against diverse versions/baselines;
- or a mixture.

KnightCap's reported success involved online play, which is a reminder that diverse opponents can generate more useful learning data than one self-play policy.

For your contest project, all training occurs offline before submission.

## Simplified modern experiment

Before implementing full eligibility traces, test:

```text
target_t = discounted / lambda-mixed future value or final result
```

using saved trajectories in PyTorch.

Then implement paper-faithful TDLeaf once the pipeline is correct.

## Search/eval separation

Never differentiate through alpha-beta itself. Treat the search's selected leaves/PV as data-generation decisions, then update the evaluator.

## Debugging

- verify terminal +1/0/-1 orientation;
- make a tiny synthetic trajectory and calculate TD(lambda) targets by hand;
- verify lambda=0 behaves like one-step TD;
- verify lambda near 1 approaches long-horizon/Monte-Carlo behavior;
- check search leaf extraction deterministically.

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


## Feasibility verdict

A good research/tuning method for a small evaluator, but less direct than supervised Stockfish distillation when strong labels are explicitly allowed.

## Papers

1. Jonathan Baxter, Andrew Tridgell, Lex Weaver, **TDLeaf(lambda): Combining Temporal Difference Learning with Game-Tree Search**: https://arxiv.org/abs/cs/9901001
2. Baxter, Tridgell, Weaver, **KnightCap: A chess program that learns by combining TD(lambda) with game-tree search**: https://arxiv.org/abs/cs/9901002
3. Giraffe: https://arxiv.org/abs/1509.01549
4. AI Chessathon: https://aichessathon.com/docs

## Coding-agent instruction

Implement TDLeaf as an evaluator-training method, not as a replacement for legal move search. Preserve a baseline evaluator and run checkpoint-vs-baseline games after every training stage.
