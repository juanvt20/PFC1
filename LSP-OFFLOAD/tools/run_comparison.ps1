$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$plan = Get-Content -Raw (Join-Path $taskRoot 'configs/short_comparison.json') | ConvertFrom-Json
$image = 'pfc-lsp-offload:torch271-cu128'

function Invoke-Measurement([string]$Model, [string]$Method, [string]$Kind, [int]$Repeat = 0) {
    $runId = "$($plan.campaign)-$Kind-$Model-$Method-$Repeat"
    if ($runId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid run ID' }
    $outDir = Join-Path $taskRoot "results/$runId"
    $logFile = Join-Path $taskRoot "logs/$runId.log"
    if (Test-Path -LiteralPath $outDir) {
        $saved = Get-Content -Raw (Join-Path $outDir 'summary.json') | ConvertFrom-Json
        if ($saved.status -in @('success','oom')) {
            Write-Host "Keeping existing evidence: $runId ($($saved.status))"
            return $saved
        }
        throw "Incomplete/failed run exists: $runId; inspect before retrying under a new campaign."
    }
    if (Test-Path -LiteralPath $logFile) { throw "Unresolved log exists: $logFile" }
    $commit = (git -C (Split-Path -Parent $taskRoot) rev-parse HEAD).Trim()
    $dirty = [bool](git -C (Split-Path -Parent $taskRoot) status --porcelain -- LSP-OFFLOAD)
    $steps = if ($Kind -eq 'pilot') { $plan.pilot_steps } else { $plan.performance_steps }
    $containerName = "pfc-lsp-$runId"
    $dockerArgs = @('run','--name',$containerName,'--pull=never','--network','none','--gpus','all',
        '--mount',"type=bind,source=$taskRoot,target=/work",
        '--mount',"type=bind,source=$taskRoot\original\upstream,target=/work/original/upstream,readonly",
        '--mount',"type=bind,source=$taskRoot\adaptado\upstream,target=/work/adaptado/upstream,readonly",
        '-e','OMP_NUM_THREADS=14','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        '-e',"PFC_COMMIT=$commit",'-e',"PFC_DIRTY=$dirty",$image,
        'python','-u','tools/run_pilot.py','--model',$Model,'--method',$Method,'--kind',$Kind,
        '--run-id',$runId,'--steps',"$steps",'--warmup',"$($plan.warmup_steps)",'--seed',"$($plan.seed)")
    $started = [DateTime]::UtcNow
    $ErrorActionPreference = 'Continue'
    & $dockerExe @dockerArgs 2>&1 | Tee-Object -FilePath $logFile | Out-Host
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $ended = [DateTime]::UtcNow
    if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    $inspection = & $dockerExe inspect $containerName
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect own run container $containerName" }
    $inspection | Set-Content -Encoding UTF8 (Join-Path $outDir 'docker-inspect.json')
    $containerState = @($inspection | ConvertFrom-Json)[0]
    $record = [ordered]@{ executable=$dockerExe; argv=$dockerArgs; exit_code=$code;
        started_utc=$started.ToString('o'); ended_utc=$ended.ToString('o');
        host_wall_seconds=($ended-$started).TotalSeconds; git_commit=$commit; git_dirty=$dirty;
        image_id=$containerState.Image; oom_killed=$containerState.State.OOMKilled;
        script_sha256=(Get-FileHash -Algorithm SHA256 (Join-Path $PSScriptRoot 'run_pilot.py')).Hash;
        log="logs/$runId.log" }
    $record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $outDir 'command.json')
    $summaryPath = Join-Path $outDir 'summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        $status = if ($containerState.State.OOMKilled) { 'oom' } else { 'failed' }
        [ordered]@{ run_id=$runId; model=$Model; method=$Method; kind=$Kind; status=$status;
            exit_code=$code; oom_killed=$containerState.State.OOMKilled;
            error='Process ended without Python summary; see Docker state, log and last samples/progress';
            host_wall_seconds=($ended-$started).TotalSeconds; convergence_claim=$false
        } | ConvertTo-Json | Set-Content -Encoding UTF8 $summaryPath
    }
    # Remove only this completed task container, without removing volumes or images.
    if ($containerState.State.Running) { throw "Unexpected still-running container: $containerName" }
    & $dockerExe rm $containerName | Out-Host
    $saved = Get-Content -Raw $summaryPath | ConvertFrom-Json
    if (($saved.status -eq 'success') -and ($code -ne 0)) { throw 'Success/exit-code mismatch' }
    return $saved
}

if ($plan.phase -eq 'pilots') {
    $model = $plan.initial_model
    $zero = Invoke-Measurement $model 'zero' 'pilot'
    if ($zero.status -eq 'failed') { throw 'Baseline pilot failed; inspect recorded evidence.' }
    $lsp = $null
    if ($zero.status -eq 'success') { $lsp = Invoke-Measurement $model 'lsp' 'pilot' }
    if ($lsp -and $lsp.status -eq 'failed') { throw 'LSP pilot failed; inspect recorded evidence.' }
    if ($zero.status -eq 'oom' -or ($lsp -and $lsp.status -eq 'oom')) {
        $model = $plan.only_oom_fallback
        $fetchScript = Join-Path $PSScriptRoot 'fetch_pilot_assets.ps1'
        & powershell.exe -NoProfile -File $fetchScript -Model $model
        if ($LASTEXITCODE -ne 0) { throw 'Fallback assets download failed' }
        $zero = Invoke-Measurement $model 'zero' 'pilot'
        if ($zero.status -ne 'success') { throw 'Medium baseline failed; no further size fallback authorized.' }
        $lsp = Invoke-Measurement $model 'lsp' 'pilot'
        if ($lsp.status -ne 'success') { throw 'Medium LSP failed; inspect recorded evidence.' }
    }
    [ordered]@{ model=$model; zero=$zero.run_id; lsp=$lsp.run_id; passed=$true } |
        ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $taskRoot "results/$($plan.campaign)-pilot-gate.json")
} elseif ($plan.phase -eq 'performance') {
    $gate = Get-Content -Raw (Join-Path $taskRoot "results/$($plan.campaign)-pilot-gate.json") | ConvertFrom-Json
    if (-not $gate.passed) { throw 'Pilot gate has not passed' }
    if (git -C (Split-Path -Parent $taskRoot) status --porcelain -- LSP-OFFLOAD) {
        throw 'Commit the adaptation/tests/scripts/config before the principal comparison.'
    }
    for ($repeat=1; $repeat -le $plan.repetitions; $repeat++) {
        $order = if ($repeat % 2 -eq 1) { @('zero','lsp') } else { @('lsp','zero') }
        foreach ($method in $order) {
            $run = Invoke-Measurement $gate.model $method 'performance' $repeat
            if ($run.status -ne 'success') { throw "Performance run failed: $($run.run_id)" }
        }
    }
} else { throw 'Unknown comparison phase' }
