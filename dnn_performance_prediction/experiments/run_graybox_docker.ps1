$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$DataRoot = "/workspace/.research/upstream/NeuSight/scripts/asplos/data/dataset"
$OutputRoot = "/workspace/survey/dnn_performance_prediction/experiments/results/graybox_calibration"
$Image = "dnn-graybox-calibration:20260810"

docker build `
    --file (Join-Path $PSScriptRoot "Dockerfile.graybox") `
    --tag $Image `
    $PSScriptRoot
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with code $LASTEXITCODE"
}

docker run --rm `
    --volume "${Workspace}:/workspace" `
    $Image `
    --data-root $DataRoot `
    --output-dir $OutputRoot `
    --seeds 10 `
    --budgets 8,16,32,64,128 `
    --selective-min-samples 8 `
    --selective-shrinkage 16
if ($LASTEXITCODE -ne 0) {
    throw "Gray-box experiment failed with code $LASTEXITCODE"
}
