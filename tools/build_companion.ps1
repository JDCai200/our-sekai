param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path $PSScriptRoot -Parent
$taskWorker = Join-Path $taskProject 'artifacts/worker/AutoChart'
if (-not (Test-Path -LiteralPath (Join-Path $taskWorker 'OurSekai.Worker.exe'))) { throw '先构建离线采音组件。' }
& $Python -m PyInstaller --noconfirm --distpath (Join-Path $taskProject 'artifacts/companion') --workpath (Join-Path $taskProject '.build-tools/companion') (Join-Path $taskProject 'companion/windows.spec')
if ($LASTEXITCODE -ne 0) { throw '配套工具界面打包失败。' }
$taskRelease = Join-Path $taskProject 'artifacts/companion/OurSekai'
$taskDestination = Join-Path $taskRelease 'AutoChart'
# Reuse validated worker files locally. A release ZIP contains ordinary copies.
Get-ChildItem -LiteralPath $taskWorker -Recurse -File | ForEach-Object {
    $taskRelative = $_.FullName.Substring($taskWorker.Length).TrimStart('\')
    $taskTarget = Join-Path $taskDestination $taskRelative
    New-Item -ItemType Directory -Path (Split-Path $taskTarget -Parent) -Force | Out-Null
    if (-not (Test-Path -LiteralPath $taskTarget)) {
        try { New-Item -ItemType HardLink -Path $taskTarget -Target $_.FullName -ErrorAction Stop | Out-Null }
        catch { Copy-Item -LiteralPath $_.FullName -Destination $taskTarget }
    }
}
Copy-Item -LiteralPath (Join-Path $taskProject 'docs/companion-user-guide.md') -Destination (Join-Path $taskRelease '使用说明.md')
Copy-Item -LiteralPath (Join-Path $taskProject 'LICENSE') -Destination (Join-Path $taskRelease 'LICENSE')
& (Join-Path $taskRelease 'OurSekai.exe') --self-test (Join-Path $taskProject 'artifacts/companion-self-test.json')
if ($LASTEXITCODE -ne 0) { throw '独立界面组件自检失败。' }
