param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference='Stop'
function Get-LocalSha256([string]$FilePath) {
    $stream=[IO.File]::OpenRead($FilePath)
    $hasher=[Security.Cryptography.SHA256]::Create()
    try {return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}
    finally {$stream.Dispose();$hasher.Dispose()}
}

$projectRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path
$runsPrefix=(Join-Path $projectRoot 'runs')+'\'
$records=Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$logPath=Join-Path (Split-Path -Parent $Manifest) 'deleted_originals.jsonl'
if (Test-Path -LiteralPath $logPath) {throw 'Deletion already recorded; inspect before retrying'}
# Validate every exact path, original hash and archive hash before deleting any file.
foreach ($record in $records) {
    $target=[IO.Path]::GetFullPath((Join-Path $projectRoot $record.path))
    $archive=[IO.Path]::GetFullPath((Join-Path $projectRoot $record.archive))
    if (-not $target.StartsWith($runsPrefix,[StringComparison]::OrdinalIgnoreCase) -or $archive -ne ($target+'.gz')) {throw 'Unsafe archive path'}
    $item=Get-Item -LiteralPath $target
    if ($item.Extension -notin @('.rst','.db','.esav','.mode') -or $item.Length -ne $record.original_bytes) {throw 'Changed original'}
    $ancestor=$item
    while ($null -ne $ancestor -and $ancestor.FullName -ne $projectRoot) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse point'}
        $ancestor=if ($ancestor -is [IO.FileInfo]) {$ancestor.Directory} else {$ancestor.Parent}
    }
    if ((Get-Item -LiteralPath $archive).Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Linked archive'}
    if ((Get-LocalSha256 $target) -ne $record.original_sha256) {throw 'Original checksum changed'}
    if ((Get-LocalSha256 $archive) -ne $record.archive_sha256) {throw 'Archive checksum changed'}
}
foreach ($record in $records) {
    $target=[IO.Path]::GetFullPath((Join-Path $projectRoot $record.path))
    Remove-Item -LiteralPath $target
    $record | ConvertTo-Json -Compress | Add-Content -LiteralPath $logPath -Encoding utf8
}
Write-Output "Removed $($records.Count) verified uncompressed checkpoint copies; lossless archives retained."
