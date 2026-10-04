param([string]$Python)
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path $PSScriptRoot -Parent
$taskBackend = Join-Path $taskProject 'backend'
if (-not $Python) {
    $taskLocal = Join-Path $taskBackend 'runtime.local.json'
    if (Test-Path -LiteralPath $taskLocal) { $Python = (Get-Content -LiteralPath $taskLocal -Raw | ConvertFrom-Json).python }
    else { $Python = Join-Path $taskBackend '.venv/Scripts/python.exe' }
}
if (-not (Test-Path -LiteralPath $Python)) { throw '请先配置 Python 3.11 采音运行环境。' }
& $Python -s -X utf8 (Join-Path $taskBackend 'worker.py') --check-models
if ($LASTEXITCODE -ne 0) { throw '离线模型预检失败。' }
& $Python -m PyInstaller --noconfirm --distpath (Join-Path $taskProject 'artifacts/worker') --workpath (Join-Path $taskProject '.build-tools/pyinstaller') (Join-Path $taskBackend 'worker.spec')
if ($LASTEXITCODE -ne 0) { throw '采音组件打包失败。' }
& $Python -s -X utf8 (Join-Path $PSScriptRoot 'complete_worker_dependencies.py') (Join-Path $taskProject 'artifacts/worker/AutoChart')
if ($LASTEXITCODE -ne 0) { throw '动态依赖补全失败。' }
& (Join-Path $taskProject 'artifacts/worker/AutoChart/OurSekai.Worker.exe') --check-models
if ($LASTEXITCODE -ne 0) { throw '打包后的采音组件预检失败。' }
