# One-time setup: creates a Python 3.11 virtual environment with CUDA PyTorch + dependencies.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (Python 3.11)..."
    py -3.11 -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
Write-Host "Installing PyTorch (CUDA 12.6)..."
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
Write-Host "Installing app dependencies..."
& $py -m pip install -r requirements.txt

Write-Host "Checking local AI (Ollama)..."
try {
    $tags = Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 5
    if (-not ($tags.models.name -contains "qwen2.5vl:7b")) {
        Write-Host "Pulling vision model qwen2.5vl:7b ..."
        ollama pull qwen2.5vl:7b
    }
} catch {
    Write-Warning "Ollama is not running. Start Ollama, then run: ollama pull qwen2.5vl:7b"
}
Write-Host "`nSetup complete. Start the app with run.bat"
