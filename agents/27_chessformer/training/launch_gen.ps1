# Launch detached self-play data generation workers (they survive the launching shell).
# Each worker is an independent process (no multiprocessing.Pool), writes one shard per
# --chunk positions, and skips shards that already exist, so re-running is safe.
#
#   powershell -File training/launch_gen.ps1 -Workers 4 -SeedBase 2006 -PerWorker 40000
#
param(
    [int]$Workers = 4,
    [int]$SeedBase = 2006,
    [int]$PerWorker = 40000,
    [int]$Chunk = 4000,
    [int]$Depth = 3
)
$py = "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $here "gen_data.py"
$data = Join-Path $here "data"
New-Item -ItemType Directory -Force $data | Out-Null
$shardsPerWorker = [math]::Ceiling($PerWorker / $Chunk)
for ($w = 0; $w -lt $Workers; $w++) {
    $seed = $SeedBase + $w * $shardsPerWorker
    $log = Join-Path $data ("gen_w{0}_s{1}.log" -f $w, $seed)
    $args = @($script, "--out", $data, "--workers", "1", "--positions", "$PerWorker", "--chunk", "$Chunk",
              "--depth", "$Depth", "--budget", "2.0", "--epsilon", "0.1", "--harvest-depth", "2",
              "--max-plies", "240", "--seed", "$seed")
    $p = Start-Process -FilePath $py -ArgumentList $args -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$', '.err')
    # keep Normal priority: BelowNormal starves completely on this 100 %-loaded box
    Write-Output ("worker {0} pid {1} seeds {2}..{3} -> {4}" -f $w, $p.Id, $seed, ($seed + $shardsPerWorker - 1), $log)
}
