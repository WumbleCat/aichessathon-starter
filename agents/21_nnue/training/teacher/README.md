# Teacher (NOT shipped)

Stockfish 17.1 (official release binary, downloaded from the Stockfish GitHub
releases) lives here purely as an OFFLINE labelling teacher for `../datagen.py`.

It is never imported, called, or packaged by `agent.py`. `harness/package.py`
ships only root `*.py` files and the `weights/` directory.
