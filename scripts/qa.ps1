$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
node scripts\qa.cjs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
