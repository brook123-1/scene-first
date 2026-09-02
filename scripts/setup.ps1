$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'

function Test-Python312 {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$PrefixArguments = @()
    )
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    & $Executable @PrefixArguments -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' *> $null
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorPreference
    return $probeExitCode -eq 0
}

if (-not (Test-Path $python)) {
    if ($env:SCENE_FIRST_PYTHON) {
        if (-not (Test-Path -LiteralPath $env:SCENE_FIRST_PYTHON) -or
            -not (Test-Python312 -Executable $env:SCENE_FIRST_PYTHON)) {
            throw 'SCENE_FIRST_PYTHON must point to a Python 3.12 executable.'
        }
        & $env:SCENE_FIRST_PYTHON -m venv (Join-Path $root '.venv')
    }
    if (-not (Test-Path $python)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher -and (Test-Python312 -Executable $launcher.Source -PrefixArguments @('-3.12'))) {
            & py -3.12 -m venv (Join-Path $root '.venv')
        }
    }
    if (-not (Test-Path $python)) {
        $systemPython = Get-Command python -ErrorAction SilentlyContinue
        if ($systemPython -and (Test-Python312 -Executable $systemPython.Source)) {
            & $systemPython.Source -m venv (Join-Path $root '.venv')
        }
    }
    if (-not (Test-Path $python)) {
        # Codex desktop can provide an isolated Python without registering it
        # in the Windows launcher. This optional fallback is discovered from
        # the current profile and is never required for a normal installation.
        $profileRoot = [Environment]::GetFolderPath('UserProfile')
        $codexPython = Join-Path $profileRoot '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if ((Test-Path -LiteralPath $codexPython) -and (Test-Python312 -Executable $codexPython)) {
            & $codexPython -m venv (Join-Path $root '.venv')
        }
    }
    if (-not (Test-Path $python)) {
        throw 'Python 3.12 was not found. Install Python 3.12, or set SCENE_FIRST_PYTHON to its executable, then run npm run app:setup again.'
    }
}

& $python -m pip install -r (Join-Path $root 'requirements.txt')
Write-Host ''
Write-Host 'Setup complete. Run: npm run app:start'
