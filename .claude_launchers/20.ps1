
Set-Location "E:\sourcecode\ai-chess-original\aichessathon-starter"

$Host.UI.RawUI.WindowTitle = "Chess Agent 20 - PVS selective-search engine"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " CLAUDE CHESS AGENT 20" -ForegroundColor Cyan
Write-Host " PVS selective-search engine" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "README:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\my-agents-readmes\20_PVS_SELECTIVE_SEARCH.md" -ForegroundColor Yellow

Write-Host ""

Write-Host "Implementation:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\agents\20_pvs" -ForegroundColor Green

Write-Host ""

$Prompt = Get-Content -Raw "E:\sourcecode\ai-chess-original\aichessathon-starter\.claude_prompts\20.txt"

claude "$Prompt"

