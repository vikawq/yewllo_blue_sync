param(
    [ValidateSet(1, 2)]
    [int]$TensorParallelSize = 1
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Repo = Join-Path $Workspace ".research\upstream\vidur"
$Python = Join-Path $Workspace ".research\envs\vidur\Scripts\python.exe"
$OutputRoot = Join-Path $PSScriptRoot "results\vidur_tp$TensorParallelSize"
$CacheRoot = Join-Path $Workspace ".research\cache\vidur_tp$TensorParallelSize"
$MatplotlibCache = Join-Path $Workspace ".research\cache\matplotlib"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Vidur Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Repo)) {
    throw "Vidur repository not found: $Repo"
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $CacheRoot, $MatplotlibCache | Out-Null
$env:WANDB_MODE = "disabled"
$env:MPLCONFIGDIR = $MatplotlibCache

$VidurArgs = @(
    "--replica_config_device", "a100",
    "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
    "--cluster_config_num_replicas", "1",
    "--replica_config_tensor_parallel_size", $TensorParallelSize.ToString(),
    "--replica_config_num_pipeline_stages", "1",
    "--request_generator_config_type", "synthetic",
    "--synthetic_request_generator_config_num_requests", "16",
    "--length_generator_config_type", "fixed",
    "--fixed_request_length_generator_config_prefill_tokens", "256",
    "--fixed_request_length_generator_config_decode_tokens", "32",
    "--interval_generator_config_type", "poisson",
    "--poisson_request_interval_generator_config_qps", "2.0",
    "--replica_scheduler_config_type", "sarathi",
    "--sarathi_scheduler_config_batch_size_cap", "64",
    "--sarathi_scheduler_config_chunk_size", "128",
    "--random_forrest_execution_time_predictor_config_k_fold_cv_splits", "2",
    "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size", "1024",
    "--random_forrest_execution_time_predictor_config_prediction_max_batch_size", "64",
    "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request", "1024",
    "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling",
    "--random_forrest_execution_time_predictor_config_num_estimators", "50",
    "--random_forrest_execution_time_predictor_config_max_depth", "8",
    "--random_forrest_execution_time_predictor_config_min_samples_split", "2",
    "--metrics_config_output_dir", $OutputRoot,
    "--metrics_config_cache_dir", $CacheRoot,
    "--metrics_config_enable_chrome_trace",
    "--no-metrics_config_store_plots"
)

Push-Location $Repo
try {
    & $Python -m vidur.main @VidurArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Vidur exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

