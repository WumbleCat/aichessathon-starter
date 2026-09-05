$readmeDir = ".\my-agents-readmes"
$outputDir = ".\my-agents"

# Make sure README directory exists
if (-not (Test-Path $readmeDir)) {
    Write-Error "README directory not found: $readmeDir"
    exit 1
}

# Make sure output directory exists
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Find algorithm READMEs, ignoring the master README.md
$readmes = Get-ChildItem "$readmeDir\*.md" |
    Where-Object { $_.Name -ne "README.md" } |
    Sort-Object Name

if ($readmes.Count -eq 0) {
    Write-Error "No algorithm README files found in $readmeDir"
    exit 1
}

Write-Host ""
Write-Host "Found $($readmes.Count) algorithm READMEs."
Write-Host ""

foreach ($readme in $readmes) {

    $algorithm = $readme.BaseName

    Write-Host "Launching Claude agent: $algorithm"

    $prompt = @"
You are implementing an independent chess bot for this repository.

YOUR SPECIFICATION:
Read @my-agents-readmes/$($readme.Name) completely before implementing anything.

FIRST INSPECT:
- agent.py
- harness/
- harness/runner.py
- harness/referee.py
- baselines/
- CLAUDE.md
- AGENTS.md

Understand exactly how agents communicate with the chess harness.

YOUR TASK:
Implement the algorithm described in:

my-agents-readmes/$($readme.Name)

Create the new bot inside:

my-agents/$algorithm/

The directory should contain whatever files are necessary for the harness
to execute the bot, including agent.py if that is the required interface.

IMPORTANT RULES:

1. Do not modify harness/.
2. Do not modify baselines/.
3. Do not modify another directory inside my-agents/.
4. Only write your implementation to:
   my-agents/$algorithm/
5. Use the existing chess libraries in the repository.
6. Always return legal moves.
7. Follow the existing agent interface exactly.
8. Do not merely explain the algorithm. Actually implement it.
9. Test the implementation.
10. Fix errors found during testing.

After implementation, run a small arena test against an existing baseline
if the repository supports it.

For example:

uv run python -m harness.arena --agent my-agents/$algorithm --opponent baselines/greedy --games 2

Adjust the command if necessary based on how the repository actually works.

When finished, report:
- files created
- algorithm implemented
- tests performed
- arena results
- known limitations
"@

    claude --bg --name $algorithm $prompt

    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "================================="
Write-Host "All $($readmes.Count) agents dispatched."
Write-Host "================================="
Write-Host ""
Write-Host "Monitor them with:"
Write-Host ""
Write-Host "    claude agents"
Write-Host ""