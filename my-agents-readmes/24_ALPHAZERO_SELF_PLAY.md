# AlphaZero-Style Self-Play Reinforcement Learning

## Goal

Implement a reduced AlphaZero-style chess system from scratch:

```text
policy/value network
+ PUCT MCTS
+ self-play
+ policy/value training
```

This is legal if your network starts from random initialization and you train it yourself, but it is a high-compute route.

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


## Original algorithm

Network:

```text
(p, v) = f_theta(s)
```

where:

- `p` is a prior distribution over actions;
- `v` predicts expected game outcome.

Self-play uses MCTS to create an improved root visit distribution `pi`.

Training target:

```text
position s_t
search policy pi_t
final result z in (-1, 0, 1)
```

Loss from the AlphaZero paper:

```text
(z - v)^2
- pi^T log(p)
+ c ||theta||^2
```

## Contest adaptation

Do NOT try to reproduce DeepMind-scale AlphaZero.

Start with:

```text
small residual CNN or compact board network
64-256 channels, shallow depth
policy head
value head
```

Then benchmark model file and CPU inference.

The final contest agent still has only one CPU core and no GPU, so reduce MCTS simulations aggressively.

## Action encoding

A faithful chess mapping commonly uses an AlphaZero-style spatial action representation. A simpler independent implementation may use a fixed indexed move vocabulary plus legal masking.

Requirements:

- all normal moves;
- promotions including underpromotions;
- castling;
- side-to-move consistency.

## Self-play loop

```text
initialize network randomly

repeat:
    generate self-play games using MCTS
    save (s, pi, z)
    sample replay buffer
    update network
    periodically evaluate new checkpoint
```

Because the paper's AlphaZero continually updates one network, you do not need an old-vs-new gating system unless you want it for stability.

## PUCT selection

Use a score with:

```text
Q(s,a) + exploration_bonus(P(s,a), N(s,a), N(s))
```

The coding agent should implement the exact chosen PUCT formula consistently and write unit tests for sign perspective.

Add Dirichlet noise to root priors during training only, not tournament inference.

## Self-play efficiency

Use training hardware outside the contest. Parallelize game generation offline if possible.

Store compact examples; self-play can explode disk usage.

Curriculum options:

- begin from standard start;
- later include random legal near-balanced positions;
- include endgame starts;
- optionally bootstrap with supervised engine-labelled training **only if you still initialize your own model**, but then describe it as hybrid supervised + RL rather than pure AlphaZero.

## Tournament inference

At each move:

```text
run limited MCTS
choose highest visit move
```

Reuse the subtree between your own turns if safely reconcilable with the opponent move and current FEN.

Because the contest grants a single CPU core, test very small simulation budgets such as:

```text
25 / 50 / 100 / 200 simulations
```

rather than assuming thousands.

## Debugging

- Check every MCTS edge value is from a documented player perspective.
- Verify terminal checkmate/draw overrides network value.
- Legal-mask policy before expansion.
- Visit counts should sum to completed simulations.
- Root Dirichlet noise disabled in evaluation.
- Replay target `z` flips perspective correctly.

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

**Research value: high. Contest risk: high.**

Original AlphaZero used massive specialized hardware; your reduced implementation may be much weaker than PVS+NNUE on one CPU. Build only after a strong classical baseline exists.

## Paper

David Silver et al., **Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm**: https://arxiv.org/abs/1712.01815

Key paper concepts to follow:
- policy/value network;
- MCTS-guided self-play;
- visit-count policy targets;
- terminal-result value targets;
- joint policy/value loss;
- random initialization.

Also see the competition constraints: https://aichessathon.com/docs

## Coding-agent instruction

Implement a small, auditable AlphaZero derivative rather than a scale replica. First prove MCTS/sign correctness on tiny toy positions, then train. Compare it directly against the PVS baseline under identical contest clocks.
