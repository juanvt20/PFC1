$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$campaignDir = Join-Path $taskRoot 'results/scheduler-v3-main-v1'
& $dockerExe inspect --format '{{json .State}}' 'pfc-lsp-scheduler-v3-main-v1'
if ($LASTEXITCODE -ne 0) { throw 'Campaign container not found; no new process was started.' }
$statusPath = Join-Path $campaignDir 'status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -Raw $statusPath | ConvertFrom-Json
    $status | Select-Object state,stage,completed,elapsed_seconds,updated_unix,exit_code,error | ConvertTo-Json -Depth 4
    if ($status.stage) {
        $progressPath = Join-Path $campaignDir "$($status.stage)/progress.json"
        if (Test-Path -LiteralPath $progressPath) { Get-Content -Raw $progressPath }
        Get-Content -LiteralPath (Join-Path $campaignDir "$($status.stage).log") -Tail 6
    }
}
Write-Output "Report directory: $campaignDir/report"
