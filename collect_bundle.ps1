# Run this from the repository root (PowerShell)
# Copies all project files and directories into gemma_project_bundle (including large binaries).

$dest = Join-Path -Path (Get-Location) -ChildPath "gemma_project_bundle"
Write-Host "Destination bundle: $dest"

# Ensure destination exists
if (-Not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }

# Copy top-level files
$files = @(
    'pipeline.py', 'app.py', 'run_all.py', 'hybrid_query_test.py',
    'requirements.txt', 'Dockerfile', 'README.md', 'rag_metadata_demo.json', 'hybrid_query_results.jsonl'
)
foreach ($f in $files) {
    if (Test-Path $f) {
        Copy-Item -Path $f -Destination $dest -Force
        Write-Host "Copied $f"
    } else {
        Write-Host "Skipping missing file: $f"
    }
}

# Use robocopy for directories (preserves files, recurses, robust for large files)
$dirs = @('data','cricket','cricket-commentary-model','cricket-fixed','tests')
foreach ($d in $dirs) {
    if (Test-Path $d) {
        $dstDir = Join-Path $dest $d
        robocopy $d $dstDir /E /NFL /NDL /NJH /NJS | Out-Null
        Write-Host "Robocopyed $d -> $dstDir"
    } else {
        Write-Host "Skipping missing directory: $d"
    }
}

Write-Host "Bundle creation complete. Check the folder: $dest"