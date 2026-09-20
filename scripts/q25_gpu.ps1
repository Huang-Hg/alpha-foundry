# Run Q25 with the workspace-local CUDA Python dependencies.
# Example: ./scripts/q25_gpu.ps1 scripts/smoke_q25_combo_gate.py
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$quantRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $quantRoot 'tmp/q25_cuda_deps'
$env:CUPY_CACHE_DIR = Join-Path $quantRoot 'tmp/cupy_cache'
$env:PYTHONUTF8 = '1'
Push-Location $quantRoot
try {
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Q25 Python exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
