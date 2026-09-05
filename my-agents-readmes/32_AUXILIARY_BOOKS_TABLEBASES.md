# Auxiliary Data: Opening Books and Endgame Tablebases

## Goal

Use legal shipped data that can complement any search/model architecture.

These are **not learning models**, but they were mentioned because the contest explicitly permits them.

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


## Opening books

The environment includes `chess.polyglot`.

Potential deployment:

```text
if current position in book:
    choose weighted book move
else:
    run engine
```

### Contest caveat

Games start from curated positions close to level, not necessarily the standard initial position. Therefore a conventional opening book may have limited coverage/value.

Do not allocate much of the 50 MB budget until measured against representative starts.

## Endgame tablebases

The environment includes `chess.syzygy`.

Use:

```text
if piece_count within shipped tablebase coverage:
    probe
    select exact WDL/DTZ-aware move
else:
    search normally
```

### Size constraint

Full Syzygy sets are vastly larger than the contest limit. Only include a tiny subset if it demonstrably produces more value than neural/search improvements.

## Important rule distinction

The docs allow opening books and endgame tablebases, but prohibit shipping a database of third-party engine moves/evaluations for runtime lookup as a disguised engine.

Therefore:
- standard permitted book/tablebase formats: allowed by docs;
- custom giant "Stockfish says play X for FEN Y" lookup: prohibited.

## Engineering priorities

Recommended order:

```text
search correctness
time manager
PVS/selective search
NNUE
then books/tablebases
```

## Tests

- missing book/tablebase must fail gracefully;
- all selected book moves are legal;
- tablebase lookup time is bounded;
- ZIP uncompressed size remains < 50 MB;
- startup remains within 90 s.

## References

1. AI Chessathon docs: https://aichessathon.com/docs
2. python-chess Polyglot docs: https://python-chess.readthedocs.io/en/latest/polyglot.html
3. python-chess Syzygy docs: https://python-chess.readthedocs.io/en/latest/syzygy.html

## Coding-agent instruction

Treat these as optional measured add-ons. Do not let static data consume the model/search budget unless arena results show a clear gain.
