param([ValidateSet('gpt2-large','gpt2-medium')][string]$Model='gpt2-large')
$ErrorActionPreference='Continue'
$taskRoot=Split-Path -Parent $PSScriptRoot
$dockerExe=Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$log=Join-Path $taskRoot "logs/assets-$Model-$stamp.log"
& $dockerExe run --rm --pull=never --mount "type=bind,source=$taskRoot,target=/work" -e HF_HUB_DISABLE_XET=1 -e HF_HUB_DISABLE_PROGRESS_BARS=1 pfc-lsp-offload:torch271-cu128 python tools/fetch_pilot_assets.py --model $Model 2>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
