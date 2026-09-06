# Submissions

Built zips, one per agent, in the layout the platform expects: `agent.py` at the root of the
archive, alongside the modules and weights it imports. Rebuild one with `harness.package` from
inside the agent's directory, which is what fixes the archive root:

```powershell
$repo = "E:\sourcecode\ai-chess-original\aichessathon-starter"
$env:PYTHONPATH = $repo
cd $repo\agents\29_deepchess
& $repo\.venv\Scripts\python.exe -m harness.package --out $repo\submissions\agent29_deepchess.zip --include "models/deepchess.npz"
cd $repo\agents\32_auxiliary
& $repo\.venv\Scripts\python.exe -m harness.package --out $repo\submissions\agent32_auxiliary.zip --include book.bin --include tables
```

| zip | unzipped | contents | measured |
|---|---|---|---|
| `agent29_deepchess.zip` | 0.93 MB | `agent.py`, `dc_engine.py`, `dc_search.py`, the v2 network | +32 =27 -41 over 100 games vs Stockfish `UCI_Elo` 2600 |
| `agent32_auxiliary.zip` | 5.13 MB | 5 modules, `book.bin`, 3/4-piece Syzygy `tables/` | +7 =38 -55 over 100 games vs the same |

Both are far inside the 50 MB cap. The `--include` list matters: only root-level `*.py` and
the paths named are packaged, so anything the agent loads at runtime has to be named. Agent
32's `tables_ext/` is deliberately left out, as its own RESULTS.md explains, and agent 29
ships one weights file rather than the whole `models/` directory, which would otherwise carry
four unused checkpoints.

Verify a zip before uploading by playing from the extracted copy, not from the source tree,
which is what catches a missing runtime file:

```powershell
Expand-Archive submissions\agent29_deepchess.zip -DestinationPath $env:TEMP\check29
& $repo\.venv\Scripts\python.exe -m harness.arena --agent $env:TEMP\check29 --opponent baselines/minimax --games 2
```

Nothing here decides acceptance: the platform validates on upload and its log is the
authority. See `benchmarks/README.md` for how the match numbers above were produced and how
to read the failure columns.
