# PVS + Transposition Table + Modern Selective Alpha-Beta

## Goal

This is the **primary classical contest engine**. Start from the tested alpha-beta engine and turn it into a Stockfish-family *architecture* without copying/porting Stockfish code.

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


## Target architecture

```text
iterative deepening
  -> aspiration window
  -> PVS / negamax alpha-beta
       -> transposition table
       -> null-move pruning
       -> futility / reverse futility pruning
       -> late-move reductions
       -> check / selective extensions
       -> move ordering
            TT move
            captures + SEE
            killers
            history
       -> quiescence search
  -> best move from last completed depth
```

## 1. Principal Variation Search (PVS)

Search the first move with a full window. Assume later moves are worse and initially search them with a null window.

```python
if first_move:
    score = -search(child, depth-1, -beta, -alpha)
else:
    score = -search(child, depth-1, -alpha-1, -alpha)
    if alpha < score < beta:
        score = -search(child, depth-1, -beta, -alpha)
```

PVS is only effective with good move ordering.

## 2. Zobrist hash + transposition table

Generate deterministic random 64-bit keys for:

```text
piece x square
side to move
castling rights
en-passant file
```

Store entries such as:

```python
(key, depth, score, flag, best_move, age)
```

Flags:

```text
EXACT
LOWERBOUND
UPPERBOUND
```

Normalize mate scores when storing/retrieving so mate distance remains correct across different plies.

## 3. Iterative deepening + aspiration windows

At depth > 1, search around previous score:

```text
previous_score ± margin
```

If it fails high/low, widen and repeat.

## 4. Quiescence search

At depth 0, do not immediately evaluate tactically unstable positions. Search forcing captures/promotions and all legal evasions when in check.

Use:

- stand-pat when not in check;
- capture ordering;
- delta pruning;
- eventually SEE.

## 5. Null-move pruning

If:

- not in check;
- sufficient depth;
- side has meaningful non-pawn material;
- position is not likely zugzwang;

make a null move and run a reduced-depth null-window search. If it still exceeds beta, prune.

Be conservative in pawn endings and zugzwang-prone endgames.

## 6. Late Move Reductions (LMR)

For late, quiet, low-priority moves:

1. search at reduced depth;
2. if score exceeds alpha, re-search at normal depth.

Reduction should grow with both depth and move index. Keep tactical/PV/check moves less reduced.

## 7. Futility pruning

At shallow non-check nodes, use static evaluation plus a conservative margin to skip quiet moves unlikely to raise alpha.

Never apply blindly around mate scores or tactical nodes.

## 8. Move ordering

Recommended staged ordering:

```text
TT/PV move
good captures (SEE >= threshold)
promotions
killer/counter moves
history-scored quiets
bad captures
```

Maintain history scores keyed by moving piece/from-to or side/from/to.

## 9. Static Exchange Evaluation (SEE)

Estimate whether a capture sequence on one square wins or loses material by repeatedly selecting the least valuable attacker. Use SEE for:

- capture ordering;
- qsearch pruning;
- shallow main-search pruning.

## Development order

Do NOT add everything at once.

```text
A. iterative deepening
B. TT
C. move ordering
D. PVS
E. qsearch
F. aspiration
G. null move
H. LMR
I. futility
J. SEE
K. tune thresholds by games
```

Run paired matches after each step. A pruning bug can gain nodes/s while losing hundreds of Elo.

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


## Search-specific debugging

Log offline:

```text
depth
score
PV
nodes
qnodes
TT hit %
beta cutoffs
null cutoffs
LMR re-search count
time
nodes/s
```

For every heuristic, have a toggle. The coding agent should be able to run A/B matches with one feature disabled.

## Papers / references

1. Judea Pearl, **SCOUT: A Simple Game-Searching Algorithm with Proven Optimal Properties** (AAAI 1980): https://cdn.aaai.org/AAAI/1980/AAAI80-041.pdf
2. Donald E. Knuth & Ronald W. Moore, **An Analysis of Alpha-Beta Pruning** (1975): https://doi.org/10.1016/0004-3702(75)90019-3
3. Albert L. Zobrist, **A New Hashing Method With Application for Game Playing** (1970): https://research.cs.wisc.edu/techreports/viewreport.php?report=88
4. Christian Donninger, **Null Move and Deep Search: Selective-Search Heuristics for Obtuse Chess Programs** (1993): https://doi.org/10.3233/ICG-1993-16304
5. Ernst A. Heinz, **Adaptive Null-Move Pruning** (1999): https://doi.org/10.3233/ICG-1999-22302
6. Don F. Beal, **A Generalised Quiescence Search Algorithm** (1990): https://doi.org/10.1016/0004-3702(90)90072-8
7. Chess Programming Wiki, SEE: https://www.chessprogramming.org/Static_Exchange_Evaluation
8. AlphaZero paper, Methods section summarizing PVS, aspiration, null move, futility, LMR, history and SEE in classical engines: https://arxiv.org/abs/1712.01815
9. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Implement these algorithms from the papers/descriptions, not by translating a third-party engine. Preserve feature flags and regression tests. Optimize only after correctness.
