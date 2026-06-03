$dst = Join-Path (Get-Location) 'gemma_space_bundle'
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Path $dst | Out-Null
$files = @('README.md','requirements.txt','space_app.py','SPACE_README.md','hybrid_search.py','hybrid_query_test.py','evaluate_rag.py','hybrid_query_results.jsonl','rag_chunk_index.faiss','rag_chunk_metadata.json','rag_metadata_demo.json','collect_bundle.ps1')
foreach ($f in $files) {
  if (Test-Path $f) {
    Copy-Item -Path $f -Destination $dst -Force
    Write-Host "Copied $f"
  } else {
    Write-Host "Skipping missing: $f"
  }
}
$zip = Join-Path (Get-Location) 'gemma_space_bundle.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $dst '*') -DestinationPath $zip -Force
Write-Host "Created $zip"
