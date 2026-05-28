# Clear partial outputs from the 4 cases interrupted by the PC restart, then re-run them.
# Keeps .DATA + INCLUDE/, deletes all Eclipse output files so eclrun regenerates cleanly.

$cases = @("NORNE_464", "NORNE_465", "NORNE_466", "NORNE_467")
$caseRoot = "D:\NORNE\cases"
$eclrun = "C:\ecl\macros\eclrun.exe"

foreach ($case in $cases) {
    $dir = Join-Path $caseRoot $case
    Write-Output "=== Clearing partial outputs for $case ==="
    # Delete every $case.* file EXCEPT .DATA (keep the deck). INCLUDE/ stays untouched.
    Get-ChildItem $dir -File -Filter "$case.*" | Where-Object { $_.Extension -ne ".DATA" } | ForEach-Object {
        Remove-Item $_.FullName -Force
        Write-Output "  removed $($_.Name)"
    }
}

Write-Output ""
Write-Output "=== Re-running the 4 cases in parallel via eclrun ==="
$jobs = @()
foreach ($case in $cases) {
    $dir = Join-Path $caseRoot $case
    $jobs += Start-Process -FilePath $eclrun -ArgumentList "eclipse", $case `
        -WorkingDirectory $dir -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $dir "rerun.log") `
        -RedirectStandardError (Join-Path $dir "rerun_err.log")
    Write-Output "  launched $case (PID $($jobs[-1].Id))"
}

Write-Output ""
Write-Output "4 Eclipse re-runs launched. They'll take ~15 min. Check with the missing-cases query later."
