# Launch the full round robin detached, so it outlives the shell that started it.
param(
    [int]$Games = 100,
    [int]$Workers = 6,
    [int]$BaseMs = 10000,
    [int]$IncrementMs = 100
)

$repo = "E:\sourcecode\ai-chess-original\aichessathon-starter"
$python = "$repo\.venv\Scripts\python.exe"
$results = "$repo\round_robin_results"

New-Item -ItemType Directory -Force -Path $results | Out-Null

$arguments = @(
    "-u", "round_robin.py",
    "--games", $Games,
    "--workers", $Workers,
    "--base-ms", $BaseMs,
    "--increment-ms", $IncrementMs
)

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -RedirectStandardOutput "$results\run.log" -RedirectStandardError "$results\run.err" -WindowStyle Hidden -PassThru

$process.Id | Set-Content -Encoding utf8 "$results\run.pid"
Write-Output "launched pid $($process.Id), log $results\run.log"
