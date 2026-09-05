
Set-Location "E:\sourcecode\ai-chess-original\aichessathon-starter"

$Host.UI.RawUI.WindowTitle = "Chess Agent 17 - rule-based engine"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " CLAUDE CHESS AGENT 17" -ForegroundColor Cyan
Write-Host " rule-based engine" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "README:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\my-agents-readmes\17_RULE_BASED_ENGINE.md" -ForegroundColor Yellow

Write-Host ""

Write-Host "Implementation:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\agents\17_rule_based" -ForegroundColor Green

Write-Host ""

$Prompt = Get-Content -Raw "E:\sourcecode\ai-chess-original\aichessathon-starter\.claude_prompts\17.txt"

claude "$Prompt"

