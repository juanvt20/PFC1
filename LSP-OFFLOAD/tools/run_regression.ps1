param(
    [Parameter(Mandatory=$true)][ValidateSet('moments','scheduler')][string]$Suite,
    [Parameter(Mandatory=$true)][ValidateSet('original','adaptado')][string]$Source,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunId
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$testFile = if ($Suite -eq 'moments') { 'validate_moment_transport.py' } else { 'validate_scheduler_boundaries.py' }
$outDir = Join-Path $taskRoot "results/$RunId"
$logFile = Join-Path $taskRoot "logs/$RunId.log"
if ((Test-Path -LiteralPath $outDir) -or (Test-Path -LiteralPath $logFile)) {
    throw 'Run already exists; choose another RunId.'
}
$dockerArgs = @('run','--rm','--pull=never','--network','none','--gpus','all',
    '--mount',"type=bind,source=$taskRoot,target=/work",
    '--mount',"type=bind,source=$taskRoot\original\upstream,target=/work/original/upstream,readonly",
    '-e','OMP_NUM_THREADS=2','pfc-lsp-offload:torch271-cu128',
    'python',"tools/$testFile",'--source',$Source,'--run-id',$RunId)
$started = [DateTime]::UtcNow.ToString('o')
$ErrorActionPreference = 'Continue' # PowerShell 5 wraps native stderr as records.
& $dockerExe @dockerArgs 2>&1 | Tee-Object -FilePath $logFile
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $outDir) {
    $record = [ordered]@{ executable=$dockerExe; argv=$dockerArgs; exit_code=$code;
        started_utc=$started; ended_utc=[DateTime]::UtcNow.ToString('o');
        test_sha256=(Get-FileHash -Algorithm SHA256 (Join-Path $PSScriptRoot $testFile)).Hash;
        log="logs/$RunId.log" }
    $record | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $outDir 'command.json')
}
exit $code
