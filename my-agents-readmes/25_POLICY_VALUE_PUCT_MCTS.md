# Small Policy/Value Network + PUCT MCTS

## Goal

Build the **inference/search half** of an AlphaZero/Lc0-style engine without requiring pure self-play training. The policy/value network may instead be trained from your own engine-labelled dataset.

This separates two questions:

1. Is PUCT MCTS competitive on the contest CPU?
2. How should the policy/value network be trained?

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


## Architecture

```text
root position
  -> network gives policy P and value V
  -> PUCT chooses edge
  -> descend tree
  -> expand new leaf with network
  -> back up value
repeat until time/simulation limit
  -> choose root move by visits
```

## Node statistics

For each action store:

```text
prior P
visit count N
total value W
mean value Q = W/N
child pointer
```

Store a node dictionary keyed by legal move/action.

## PUCT

Use:

```text
score = Q + U
```

with an exploration term based on prior probability and visit counts. Document exactly whether `Q` is stored from parent/current-player perspective.

Do not mix sign conventions between selection and backup.

## Network

For this contest, use a **small** model:

- compact CNN;
- tiny residual network;
- Chessformer-like model only as a separate experiment;
- outputs policy logits + scalar value or WDL.

Train from scratch.

### Supervised teacher option

For each position:
- value = teacher score/WDL;
- policy = distribution derived from scores of legal moves.

This is explicitly allowed as offline training.

## Batched inference problem

Lc0 benefits heavily from GPU batching. Here there is no GPU and one CPU.

Therefore test:
- batch size 1;
- very small batches accumulated from multiple leaves only if tree algorithm supports it;
- ONNX vs PyTorch CPU;
- compact network width.

The model that evaluates fastest may win despite worse offline loss.

## Tree reuse

After selecting your move, preserve the chosen child tree. When the next FEN arrives:

1. identify opponent move by comparing board states or search child keys;
2. advance root to matching descendant;
3. discard unrelated branches.

If reconciliation fails, rebuild safely.

## Time manager

Use both:
- a soft simulation target;
- a hard wall-clock deadline.

Always keep a legal fallback move.

## Debug tests

Tiny forced positions are essential:

- mate in 1 must dominate;
- terminal node never calls network;
- a forced loss value backs up with correct signs;
- increasing simulations should stabilize rather than oscillate wildly due to a sign error.

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


## References

1. AlphaZero paper for policy/value MCTS: https://arxiv.org/abs/1712.01815
2. Lc0 developer overview for modern policy/value MCTS context: https://lczero.org/dev/overview/
3. Chessformer paper for a modern neural architecture integrated into chess search: https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d167db04a90885ad5208fe8b273668b-Abstract-Conference.html
4. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Treat PUCT MCTS as an experimentally competing search algorithm, not an assumed upgrade. The decisive metric is Elo on one CPU at 120+0.5, not neural-network accuracy.
