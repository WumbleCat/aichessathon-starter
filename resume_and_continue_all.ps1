$root = "E:\sourcecode\ai-chess-original\aichessathon-starter"
$python = "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe"

Set-Location $root

for ($i = 17; $i -le 32; $i++) {

    $sessionName = "chess-$i"

    $promptLines = @(
        "The previous run was interrupted because the Claude usage/token limit was reached.",
        "",
        "Continue EXACTLY from where you left off.",
        "",
        "Do not restart the implementation from scratch.",
        "",
        "Before doing new work:",
        "1. Review the previous conversation context.",
        "2. Inspect git status and git diff.",
        "3. Inspect the files you already created or modified.",
        "4. Check existing TODOs and progress notes.",
        "5. Check the last test/harness results.",
        "6. Determine the exact last completed step.",
        "",
        "Use this Python interpreter for ALL Python commands:",
        $python,
        "",
        "Do NOT use system Python, Conda, uv Python, or another virtual environment.",
        "",
        "When running Python commands, explicitly use:",
        "& `"$python`" <arguments>",
        "",
        "Continue implementing, testing, debugging, and improving the chess bot.",
        "Do NOT undo working code simply because the previous session was interrupted.",
        "Continue from the previous checkpoint until the README task is complete."
    )

    $prompt = $promptLines -join [Environment]::NewLine

    Write-Host ""
    Write-Host "============================================"
    Write-Host "Resuming Claude agent: $sessionName"
    Write-Host "============================================"

    claude --bg --resume $sessionName $prompt

    Start-Sleep -Milliseconds 750
}

Write-Host ""
Write-Host "All Claude agents have been resumed."
Write-Host ""
Write-Host "Run:"
Write-Host "    claude agents"
Write-Host ""
Write-Host "to view the running agents."