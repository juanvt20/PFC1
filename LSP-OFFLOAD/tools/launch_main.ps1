param([ValidatePattern('^main_experiment_v[0-9]+\.json$')][string]$Config='main_experiment_v2.json')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $taskRoot
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$planPath = Join-Path $taskRoot "configs/$Config"
$plan = Get-Content -Raw $planPath | ConvertFrom-Json
if ($plan.campaign -notmatch '^[a-z0-9-]+$') { throw 'Invalid campaign name' }
$name = "pfc-lsp-$($plan.campaign)"
$campaignDir = Join-Path $taskRoot "results/$($plan.campaign)"
if (git -C $repoRoot status --porcelain -- LSP-OFFLOAD) { throw 'Commit the exact configuration and code before launching.' }
$existing = & $dockerExe ps -a --filter "name=^/$name$" --format '{{.ID}}'
if ($LASTEXITCODE -ne 0) { throw 'Cannot query Docker' }
if ($existing) { throw "Campaign container already exists. Use tools/status_main.ps1; no duplicate launched." }
$running = & $dockerExe ps --filter 'name=pfc-lsp-scheduler-v3-main-' --format '{{.Names}}'
if ($LASTEXITCODE -ne 0) { throw 'Cannot query running campaigns' }
if ($running) { throw "Another main campaign is running: $running. No duplicate launched." }
if (Test-Path -LiteralPath $campaignDir) { throw 'Campaign output already exists. No overwrite or automatic resume.' }
foreach ($property in $plan.immutable_reference_hashes.PSObject.Properties) {
    $actual = (Get-FileHash -Algorithm SHA256 (Join-Path $taskRoot $property.Name)).Hash.ToLowerInvariant()
    if ($actual -ne $property.Value) { throw "Validated reference changed: $($property.Name)" }
}
New-Item -ItemType Directory -Path $campaignDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $campaignDir 'code') | Out-Null
Copy-Item -LiteralPath $planPath -Destination (Join-Path $campaignDir 'plan.json')
Copy-Item -LiteralPath (Join-Path $taskRoot 'configs/comparison_summary.json') -Destination (Join-Path $campaignDir 'reused-short-comparison.json')
$hashes = [ordered]@{}
foreach ($script in @('run_main.py','main_coordinator.py','main_initial_reuse.py','report_main.py')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $script) -Destination (Join-Path $campaignDir "code/$script")
    $hashes[$script] = (Get-FileHash -Algorithm SHA256 (Join-Path $campaignDir "code/$script")).Hash.ToLowerInvariant()
}
$inside = "/work/results/$($plan.campaign)"
$dockerArgs = @('run','-d','--init','--name',$name,'--restart=no','--pull=never','--network','none','--gpus','all',
    '--mount',"type=bind,source=$taskRoot,target=/work",
    '--mount',"type=bind,source=$taskRoot\original\upstream,target=/work/original/upstream,readonly",
    '--mount',"type=bind,source=$taskRoot\adaptado\upstream,target=/work/adaptado/upstream,readonly",
    '--mount',"type=bind,source=$taskRoot\models,target=/work/models,readonly",
    '--mount',"type=bind,source=$taskRoot\datasets,target=/work/datasets,readonly",
    '--mount',"type=bind,source=$taskRoot\configs,target=/work/configs,readonly",
    '--mount',"type=bind,source=$taskRoot\tools,target=/work/tools,readonly",
    '--mount',"type=bind,source=$taskRoot\results\scheduler-v3-main-v1,target=/work/results/scheduler-v3-main-v1,readonly",
    '-e','OMP_NUM_THREADS=14','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
    $plan.image,'python','-u',"$inside/code/main_coordinator.py",'--campaign-dir',$inside)
$launch = [ordered]@{ campaign=$plan.campaign; executable=$dockerExe; argv=$dockerArgs;
    commit=(git -C $repoRoot rev-parse HEAD).Trim(); git_dirty=$false;
    launched_utc=[DateTime]::UtcNow.ToString('o'); image=$plan.image; script_hashes=$hashes;
    config_sha256=(Get-FileHash -Algorithm SHA256 $planPath).Hash.ToLowerInvariant();
    detached=$true; exact_resume_supported=$false; status_command="& ./LSP-OFFLOAD/tools/status_main.ps1 -Campaign $($plan.campaign)" }
$launch | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 (Join-Path $campaignDir 'launch.json')
$containerId = & $dockerExe @dockerArgs
if ($LASTEXITCODE -ne 0) { throw 'Detached launch failed; keep launch evidence and inspect before retrying.' }
$containerId | Set-Content -Encoding ASCII (Join-Path $campaignDir 'container.id')
Write-Output "Detached campaign started: $name"
Write-Output "Status: & ./LSP-OFFLOAD/tools/status_main.ps1 -Campaign $($plan.campaign)"
Write-Output "Artifacts: $campaignDir"
