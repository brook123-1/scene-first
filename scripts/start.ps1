$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Environment missing. Run npm run app:setup first.'
}
Set-Location $root
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
