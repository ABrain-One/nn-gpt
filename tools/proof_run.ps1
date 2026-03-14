$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "=== REPO ==="
Write-Host "[PWD] $(Get-Location)"

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "[VENV] Activated .venv"
} else {
    Write-Host "[VENV] No .venv activation script found; using active Python"
}

Write-Host "=== PYTHON ==="
python -c "import sys; print('[PYTHON] ' + sys.executable)"

Write-Host "=== INSTALL ==="
python -m pip install -e ..\nn-dataset --no-deps

Write-Host "=== LOCAL IMPORT ==="
python -c "import ab.nn.api as lemur; print('[LEMUR] ' + lemur.__file__)"

Write-Host "=== PIP SHOW ==="
python -m pip show nn-dataset

Write-Host "=== RUN TEST ==="
python test.py 2>&1 | Tee-Object -FilePath test_output.txt

Write-Host "=== LAST 40 LINES ==="
Get-Content .\test_output.txt -Tail 40

Write-Host "=== PASS/FAIL ==="
$pass = Select-String -Path .\test_output.txt -Pattern "ALL TESTS PASSED" -SimpleMatch
$errors = Select-String -Path .\test_output.txt -Pattern "Traceback|ERROR"

if ($pass) {
    Write-Host "[PASS] ALL TESTS PASSED found"
} else {
    Write-Host "[FAIL] ALL TESTS PASSED not found"
}

if ($errors) {
    Write-Host "[ERROR-MATCHES]"
    $errors | ForEach-Object { $_.Line }
} else {
    Write-Host "[ERROR-MATCHES] none"
}
