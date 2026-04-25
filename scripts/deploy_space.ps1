param(
    [Parameter(Mandatory = $true)][string]$SpaceRepoUrl,
    [Parameter(Mandatory = $true)][string]$SpaceUrl,
    [string]$WorkDir = ".hf_space_tmp"
)

$ErrorActionPreference = "Stop"

if (Test-Path $WorkDir) {
    Remove-Item -Recurse -Force $WorkDir
}

git clone $SpaceRepoUrl $WorkDir

$copyList = @(
    "Dockerfile",
    "server.py",
    "inference.py",
    "openenv.yaml",
    "requirements.txt",
    "README.md",
    "env",
    "tests"
)

foreach ($item in $copyList) {
    if (Test-Path $item) {
        Copy-Item -Recurse -Force $item (Join-Path $WorkDir $item)
    }
}

Push-Location $WorkDir
git add .
git commit -m "Deploy DeepSentinel Space"
git push
Pop-Location

Write-Host "Waiting 30s for Space startup..."
Start-Sleep -Seconds 30

try {
    Write-Host "Health:" (Invoke-RestMethod -Method Get -Uri "$SpaceUrl/health" -TimeoutSec 60 | ConvertTo-Json -Compress)
} catch {
    Write-Warning "Health check failed: $($_.Exception.Message)"
}

try {
    Write-Host "Tasks:" (Invoke-RestMethod -Method Get -Uri "$SpaceUrl/tasks" -TimeoutSec 60 | ConvertTo-Json -Compress)
} catch {
    Write-Warning "Tasks check failed: $($_.Exception.Message)"
}
