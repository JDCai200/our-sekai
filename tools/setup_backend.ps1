param([string]$Python = 'python', [string]$ModelPack, [string]$FFmpegDirectory)
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path $PSScriptRoot -Parent
$taskBackend = Join-Path $taskProject 'backend'
& $Python -m venv (Join-Path $taskBackend '.venv')
if ($LASTEXITCODE -ne 0) { throw '创建 Python 3.11 环境失败。' }
$taskPython = Join-Path $taskBackend '.venv/Scripts/python.exe'
& $taskPython -m pip install -r (Join-Path $taskBackend 'requirements.txt') pyinstaller==6.10.0 pefile==2023.2.7
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
if ($ModelPack) {
    & $taskPython (Join-Path $PSScriptRoot 'restore_models.py') $ModelPack
    if ($LASTEXITCODE -ne 0) { throw '模型包恢复失败。' }
}
if ($FFmpegDirectory) {
    New-Item -ItemType Directory -Force -Path (Join-Path $taskBackend 'bin') | Out-Null
    Copy-Item -Path (Join-Path $FFmpegDirectory '*.dll') -Destination (Join-Path $taskBackend 'bin')
    Copy-Item -LiteralPath (Join-Path $FFmpegDirectory 'ffmpeg.exe'),(Join-Path $FFmpegDirectory 'LICENSE') -Destination (Join-Path $taskBackend 'bin')
}
$taskConfig = @{python=$taskPython} | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $taskBackend 'runtime.local.json'), $taskConfig)
& $taskPython (Join-Path $taskBackend 'worker.py') --check-models
if ($LASTEXITCODE -ne 0) { throw '模型配置未完成；请提供本项目的模型包。' }
