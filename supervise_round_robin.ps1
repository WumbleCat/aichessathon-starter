# Keep the round robin running until every game is played.
#
# The driver resumes from round_robin_results\games.jsonl, so relaunching after a crash, an
# out-of-memory kill or a reboot costs only the games that were in flight. Run this detached:
#
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#       '-File','supervise_round_robin.ps1' -WindowStyle Hidden
param(
    [int]$Games = 100,
    [int]$Workers = 9,
    [int]$BaseMs = 10000,
    [int]$IncrementMs = 100,
    [int]$PollSeconds = 120
)

$repo = "E:\sourcecode\ai-chess-original\aichessathon-starter"
$python = "$repo\.venv\Scripts\python.exe"
$results = "$repo\round_robin_results"
$supervisorLog = "$results\supervisor.log"

New-Item -ItemType Directory -Force -Path $results | Out-Null

$agentCount = (Get-ChildItem "$repo\agents" -Directory | Where-Object { Test-Path "$($_.FullName)\agent.py" }).Count
$target = $agentCount * ($agentCount - 1) / 2 * $Games

function Write-Log($message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message" | Add-Content -Encoding utf8 $supervisorLog
}

function Get-Played {
    if (Test-Path "$results\games.jsonl") {
        return (Get-Content "$results\games.jsonl" -ReadCount 0 | Measure-Object -Line).Lines
    }
    return 0
}

function Get-Driver {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*round_robin.py*' -and $_.CommandLine -notlike '*spawn_main*' }
}

Write-Log "supervisor up, target $target games, $Workers workers"

while ($true) {
    $played = Get-Played
    if ($played -ge $target) {
        Write-Log "all $played games played, supervisor done"
        break
    }

    if (-not (Get-Driver)) {
        # a driver that died leaves agent processes with no parent; clear them before restarting
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like '*harness*runner.py*' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

        $arguments = @(
            "-u", "round_robin.py",
            "--games", $Games,
            "--workers", $Workers,
            "--base-ms", $BaseMs,
            "--increment-ms", $IncrementMs
        )
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -RedirectStandardOutput "$results\run_$stamp.log" -RedirectStandardError "$results\run_$stamp.err" -WindowStyle Hidden -PassThru
        $process.Id | Set-Content -Encoding utf8 "$results\run.pid"
        Write-Log "started driver pid $($process.Id) at $played/$target games, log run_$stamp.log"
    }

    Start-Sleep -Seconds $PollSeconds
}
