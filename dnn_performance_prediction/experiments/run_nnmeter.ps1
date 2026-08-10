$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Repo = Join-Path $Workspace ".research\upstream\nn-Meter"
$Cli = Join-Path $Workspace ".research\envs\nnmeter-compat\Scripts\nn-meter.exe"
$Model = Join-Path $Repo "material\testmodels\mobilenetv3small_0.json"

if (-not (Test-Path -LiteralPath $Cli)) {
    throw "nn-Meter compatibility environment not found: $Cli"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "nn-Meter sample model not found: $Model"
}

Push-Location $Repo
try {
    & $Cli predict `
        --predictor cortexA76cpu_tflite21 `
        --predictor-version 1.0 `
        --nn-meter-ir $Model
    if ($LASTEXITCODE -ne 0) {
        throw "nn-Meter exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

