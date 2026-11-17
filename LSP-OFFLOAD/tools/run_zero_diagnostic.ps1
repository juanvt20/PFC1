param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9-]+$')][string]$RunId,
    [ValidateSet('baseline','candidate')][string]$Variant='baseline',
    [ValidateSet('replay','order-test')][string]$Suite='replay',
    [int]$TraceFrom=150,
    [int]$MaxSteps=166
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$dockerExe=Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$outDir=Join-Path $taskRoot "results/$RunId"
$log=Join-Path $taskRoot "logs/$RunId.log"
if ((Test-Path -LiteralPath $outDir) -or (Test-Path -LiteralPath $log)) { throw 'Diagnostic ID exists; preserve it and use a new ID.' }
$name="pfc-lsp-$RunId"
$test=if ($Suite -eq 'replay') { 'diagnose_zero.py' } else { 'test_update_event_order.py' }
$argsDocker=@('run','--name',$name,'--pull=never','--network','none','--gpus','all',
    '--mount',"type=bind,source=$taskRoot,target=/work",
    '--mount',"type=bind,source=$taskRoot\results\scheduler-v3-main-v1,target=/work/results/scheduler-v3-main-v1,readonly",
    '--mount',"type=bind,source=$taskRoot\original\upstream,target=/work/original/upstream,readonly",
    '--mount',"type=bind,source=$taskRoot\adaptado\upstream,target=/work/adaptado/upstream,readonly",
    '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','OMP_NUM_THREADS=14',
    'sha256:d5f1daf0b2d5103e7fb66bb44405839c51d0302fb8d7630fd7ed5e64497ed006',
    'python','-u',"tools/$test",'--run-id',$RunId)
if ($Suite -eq 'replay') { $argsDocker+=@('--variant',$Variant,'--trace-from',"$TraceFrom",'--max-steps',"$MaxSteps") }
$start=[DateTime]::UtcNow
$ErrorActionPreference='Continue'
& $dockerExe @argsDocker 2>&1 | Tee-Object -FilePath $log
$code=$LASTEXITCODE
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $outDir) {
    & $dockerExe inspect $name | Set-Content -Encoding UTF8 (Join-Path $outDir 'docker-inspect.json')
    [ordered]@{argv=$argsDocker;exit_code=$code;started_utc=$start.ToString('o');ended_utc=[DateTime]::UtcNow.ToString('o');
        test_sha256=(Get-FileHash -Algorithm SHA256 (Join-Path $PSScriptRoot $test)).Hash;log="logs/$RunId.log"} |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $outDir 'command.json')
}
& $dockerExe rm $name
exit $code
