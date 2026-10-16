param([Parameter(Mandatory=$true)][string]$Receipt)
$ErrorActionPreference='Stop'
$data=Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
$rootPath=(Resolve-Path -LiteralPath $data.root).Path
$allowedRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'runs')).Path
if (-not $rootPath.StartsWith($allowedRoot+'\')) {throw 'Outside runs'}
foreach ($target in $data.targets) {
    $p=[IO.Path]::GetFullPath($target)
    if (-not $p.StartsWith($rootPath+'\')) {throw 'Outside current run'}
    $item=Get-Item -LiteralPath $p
    if ($item.Extension -notin @('.out','.bat','.full','.rst','.db','.esav','.mode')) {throw 'Unexpected extension'}
    $ancestor=$item
    while ($null -ne $ancestor -and $ancestor.FullName -ne $rootPath) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse point'}
        $ancestor=if ($ancestor -is [IO.FileInfo]) {$ancestor.Directory} else {$ancestor.Parent}
    }
}
foreach ($target in $data.targets) {Remove-Item -LiteralPath $target}
$data | ConvertTo-Json -Depth 6 -Compress | Add-Content -LiteralPath (Join-Path $rootPath 'retention.jsonl')
