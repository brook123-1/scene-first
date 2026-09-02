param(
    [string]$Destination = "",
    [string]$SourceRef = "HEAD"
)

$ErrorActionPreference = "Stop"

function Get-PublicRelativePath {
    param(
        [string]$Root,
        [string]$FullName
    )
    $rootPrefix = $Root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $FullName.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Exported path escaped destination root: $FullName"
    }
    return $FullName.Substring($rootPrefix.Length).Replace('\', '/')
}

$sourceRoot = (& git rev-parse --show-toplevel).Trim()
if (-not $sourceRoot) {
    throw "Run this script from inside the private source repository."
}
$sourceRoot = [System.IO.Path]::GetFullPath($sourceRoot)

if (-not $Destination) {
    $Destination = Join-Path (Split-Path -Parent $sourceRoot) "scene-first-public"
}
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
if ($destinationRoot.TrimEnd('\', '/') -eq $sourceRoot.TrimEnd('\', '/')) {
    throw "The public export must be outside the private source repository."
}
if (Test-Path -LiteralPath $destinationRoot) {
    if (@(Get-ChildItem -LiteralPath $destinationRoot -Force).Count -gt 0) {
        throw "Destination already exists and is not empty: $destinationRoot"
    }
} else {
    New-Item -ItemType Directory -Path $destinationRoot | Out-Null
}

$resolvedRef = (& git rev-parse --verify "$SourceRef^{commit}").Trim()
if (-not $resolvedRef) {
    throw "Source ref does not resolve to a commit: $SourceRef"
}

$manifestPath = "scripts/public-export-files.txt"
$manifestText = & git show "${resolvedRef}:$manifestPath"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read $manifestPath from $resolvedRef"
}
$publicFiles = @(
    $manifestText |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith("#") } |
        Sort-Object -Unique
)
if ($publicFiles.Count -eq 0) {
    throw "The public export allowlist is empty."
}

$forbiddenPaths = @(
    '(^|/)\.git($|/)',
    '(^|/)\.env\.local$',
    '(^|/)\.local($|/)',
    '(^|/)records($|/)',
    '(^|/)\.learnings($|/)',
    '(^|/)AGENTS\.md$',
    'feishu',
    'round-neutral-01',
    '(^|/)benchmarks($|/)',
    '(^|/)debug($|/)',
    'OPEN_SOURCE_READINESS_REPORT',
    'OPEN_SOURCE_LAUNCH_CHECKLIST'
)
foreach ($relativePath in $publicFiles) {
    if ($relativePath.Contains('\') -or $relativePath.StartsWith('/') -or $relativePath.Contains('../')) {
        throw "Unsafe allowlist path: $relativePath"
    }
    foreach ($pattern in $forbiddenPaths) {
        if ($relativePath -match $pattern) {
            throw "Forbidden path in public allowlist: $relativePath"
        }
    }
    & git cat-file -e "${resolvedRef}:$relativePath"
    if ($LASTEXITCODE -ne 0) {
        throw "Allowlisted path is missing from ${resolvedRef}: $relativePath"
    }
}

$temporaryArchive = Join-Path ([System.IO.Path]::GetTempPath()) ("scene-first-public-" + [guid]::NewGuid().ToString("N") + ".zip")
try {
    & git archive --format=zip "--output=$temporaryArchive" $resolvedRef -- @publicFiles
    if ($LASTEXITCODE -ne 0) {
        throw "git archive failed for $resolvedRef"
    }
    Expand-Archive -LiteralPath $temporaryArchive -DestinationPath $destinationRoot
} finally {
    if (Test-Path -LiteralPath $temporaryArchive) {
        Remove-Item -LiteralPath $temporaryArchive -Force
    }
}

$templateRoot = Join-Path $destinationRoot "scripts/public-export"
Copy-Item -LiteralPath (Join-Path $templateRoot "gitignore.txt") -Destination (Join-Path $destinationRoot ".gitignore")
Copy-Item -LiteralPath (Join-Path $templateRoot "dockerignore.txt") -Destination (Join-Path $destinationRoot ".dockerignore")

$actualFiles = @(
    Get-ChildItem -LiteralPath $destinationRoot -Recurse -Force -File |
        ForEach-Object { Get-PublicRelativePath -Root $destinationRoot -FullName $_.FullName } |
        Sort-Object -Unique
)
foreach ($relativePath in $actualFiles) {
    foreach ($pattern in $forbiddenPaths) {
        if ($relativePath -match $pattern) {
            throw "Forbidden path present after export: $relativePath"
        }
    }
}

$expectedFiles = @($publicFiles + '.gitignore' + '.dockerignore' | Sort-Object -Unique)
$unexpected = @(Compare-Object -ReferenceObject $expectedFiles -DifferenceObject $actualFiles | Where-Object SideIndicator -eq '=>')
$missing = @(Compare-Object -ReferenceObject $expectedFiles -DifferenceObject $actualFiles | Where-Object SideIndicator -eq '<=')
if ($unexpected.Count -or $missing.Count) {
    $details = @(
        $unexpected | ForEach-Object { "unexpected: $($_.InputObject)" }
        $missing | ForEach-Object { "missing: $($_.InputObject)" }
    ) -join [Environment]::NewLine
    throw "Public export does not match the allowlist:$([Environment]::NewLine)$details"
}

$byteCount = ($actualFiles | ForEach-Object { (Get-Item -LiteralPath (Join-Path $destinationRoot $_)).Length } | Measure-Object -Sum).Sum
Write-Output "source_commit=$resolvedRef"
Write-Output "public_export=$destinationRoot"
Write-Output "file_count=$($actualFiles.Count)"
Write-Output "byte_count=$byteCount"
Write-Output "git_directory_present=$([bool](Test-Path -LiteralPath (Join-Path $destinationRoot '.git')))"
