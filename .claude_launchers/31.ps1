
Set-Location "E:\sourcecode\ai-chess-original\aichessathon-starter"

$Host.UI.RawUI.WindowTitle = "Chess Agent 31 - TDLeaf KnightCap engine"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " CLAUDE CHESS AGENT 31" -ForegroundColor Cyan
Write-Host " TDLeaf KnightCap engine" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "README:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\my-agents-readmes\31_TDLEAF_KNIGHTCAP.md" -ForegroundColor Yellow

Write-Host ""

Write-Host "Implementation:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\agents\31_tdleaf" -ForegroundColor Green

Write-Host ""

$Prompt = Get-Content -Raw "E:\sourcecode\ai-chess-original\aichessathon-starter\.claude_prompts\31.txt"

claude "$Prompt"

