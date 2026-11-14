param(
    [ValidateSet('install', 'extract', 'version')][string]$Action = 'version',
    [string]$FileName = 'LSP-OFFLOAD.pdf',
    [string]$Pages = ''
)
$ErrorActionPreference = 'Stop'
$lspRoot = Split-Path -Parent $PSScriptRoot
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
$pdfBase = 'pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel'
$dockerArgs = @('run', '--rm', '--pull=never', '--mount', "type=bind,source=$lspRoot,target=/work", '-w', '/work')
if ($Action -ne 'install') { $dockerArgs += @('--network', 'none') }
$dockerArgs += $pdfBase
switch ($Action) {
    'install' {
        $dockerArgs += @('bash', '-lc', 'python -m venv /work/.venv-pdf-linux && /work/.venv-pdf-linux/bin/python -m pip install --only-binary=:all: PyMuPDF==1.26.7')
    }
    'extract' {
        $dockerArgs += @('.venv-pdf-linux/bin/python', 'tools/pdf_extract.py', $FileName)
        if ($Pages) { $dockerArgs += @('--pages', $Pages) }
    }
    'version' {
        $dockerArgs += @('.venv-pdf-linux/bin/python', '-m', 'pip', 'show', 'PyMuPDF')
    }
}
& $dockerExe @dockerArgs
exit $LASTEXITCODE
