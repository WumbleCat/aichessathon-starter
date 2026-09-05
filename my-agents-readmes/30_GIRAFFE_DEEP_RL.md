# Giraffe-Style Learned Evaluation + Search

## Goal

Study and reproduce the central idea from Matthew Lai's Giraffe: use learning to discover a chess evaluation representation, then combine it with game-tree search.

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


## Why this matters historically

Giraffe showed that a chess engine could learn substantial evaluation knowledge and pattern recognition instead of relying entirely on hand-designed feature terms.

For a modern contest implementation, treat Giraffe as an ancestor of:

```text
learned value function + alpha-beta
```

rather than as a specification you must copy layer-for-layer.

## Build plan

### Stage 1: feature representation

Represent:
- global state;
- piece lists / square features;
- side to move;
- castling;
- material and positional structure.

A modern adaptation can use:
- dense piece/square features;
- small MLP;
- or a sparse representation similar in spirit to NNUE.

### Stage 2: supervised pretraining

Before RL, stabilize the model using:
- game outcomes;
- optionally teacher scores, if you choose a hybrid.

### Stage 3: temporal-difference / self-play refinement

Play games with your own search engine.

For states along a trajectory, update the value predictor toward later/bootstrap values and final outcomes.

Maintain a replay buffer to reduce catastrophic oscillation.

### Stage 4: alpha-beta integration

Use the trained value at quiet search leaves.

Keep terminal/mate scores exact and external to the network.

## Important modern adaptation

If your goal is Elo rather than historical replication, compare the same data under:

```text
Giraffe-like dense evaluator
vs
NNUE sparse evaluator
```

The sparse incremental network is likely much faster on contest CPU.

## Self-play training

A practical loop:

```text
current evaluator + search -> games
games -> state/result examples
train value network
checkpoint
arena current vs previous
repeat
```

Optionally mix teacher-labelled data to prevent collapse.

## Debugging

- monitor value calibration by game phase;
- verify values flip correctly under side-to-move perspective;
- watch self-play diversity;
- reject checkpoints that lower arena Elo;
- never let the network override terminal rules.

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

Historically significant and potentially workable, but for this contest use its **learned-evaluator principle** rather than faithfully rebuilding a slower 2015 architecture.

## Paper

Matthew Lai, **Giraffe: Using Deep Reinforcement Learning to Play Chess** (2015): https://arxiv.org/abs/1509.01549

Related:
- TDLeaf/KnightCap: https://arxiv.org/abs/cs/9901001
- NNUE: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- AI Chessathon: https://aichessathon.com/docs

## Coding-agent instruction

Build this only as a controlled learned-value experiment. Keep search identical while swapping evaluators, so you can tell whether learning actually adds Elo.
