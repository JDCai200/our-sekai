param([Parameter(Mandatory=$true)][string]$UnityEditor, [ValidateSet('Windows','Android')][string]$Target = 'Windows')
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path -LiteralPath $UnityEditor)) { throw '找不到 Unity 编辑器。' }
New-Item -ItemType Directory -Force -Path (Join-Path $taskProject 'artifacts') | Out-Null
$taskLog = Join-Path $taskProject "artifacts/unity-$Target.log"
$taskBuildTarget = if ($Target -eq 'Windows') { 'Win64' } else { 'Android' }
& $UnityEditor -batchmode -quit -buildTarget $taskBuildTarget -projectPath $taskProject -executeMethod "OurSekai.EditorTools.OurSekaiBuild.$Target" -logFile $taskLog
if ($LASTEXITCODE -ne 0) { throw "Unity 构建未通过，请查看 $taskLog" }
