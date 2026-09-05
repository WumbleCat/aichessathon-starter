@AGENTS.md

## Python Environment

This repository must use the following Python interpreter:

E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe

For ALL Python commands, tests, scripts, and harness executions, use this interpreter explicitly.

Do NOT use:

- `python`
- `python3`
- `py`
- `uv run python`

Examples:

```powershell
& "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe" -m pytest

& "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe" -m harness.arena --agent my-agents/01_negamax --opponent baselines/greedy --games 10
```

## Git workflow

Follow the `git-branch-workflow` skill (installed at `~/.claude/skills/git-branch-workflow`):

- Small changes (a fix, a doc edit, a script, a config tweak) are committed straight to `main`, one logical change per commit, with a `type(scope): summary` message.
- Large changes (a new agent directory, a new harness feature, a rewrite) get a `feature/<name>` branch, small commits as the work progresses, then a `--no-ff` merge into `main`.
- Never commit training data, engine binaries, or torch checkpoints; `.gitignore` lists what stays out. Runtime weights (`.npz`, `.onnx`) are tracked.
