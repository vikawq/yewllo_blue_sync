#!/usr/bin/env python3
"""Evaluate direct fitting vs. roofline-constrained selective calibration.

The experiment uses NeuSight's published, measured FP32 linear/GEMM dataset.
Source training devices and target calibration/evaluation rows are kept disjoint.
Target test shapes do not occur in the source training file.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


SOURCE_DEVICES = [
    "Tesla P100-PCIE-16GB",
    "Tesla P4",
    "Tesla T4",
    "Tesla V100-PCIE-32GB",
    "NVIDIA A100-PCIE-40GB",
]

DEFAULT_TARGET_DEVICES = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA L4",
]

FEATURE_COLUMNS = [
    "log_B",
    "log_M",
    "log_N",
    "log_K",
    "log_grid_x",
    "log_grid_y",
    "log_grid_z",
    "log_block_x",
    "log_tile_1",
    "log_tile_2",
    "log_blocks",
    "log_waves",
    "log_flops",
    "log_bytes",
    "log_ai",
    "log_ridge",
    "log_ai_over_ridge",
    "log_compute_floor_ms",
    "log_memory_floor_ms",
    "log_roofline_ms",
    "log_peak_gflops",
    "log_mem_bw",
    "log_num_sm",
    "log_l2_cache",
    "log_device_mem",
    "regime_compute",
    "path_tensor",
    "path_cutlass_simt",
    "path_sliced",
    "path_library_sgemm",
]


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: int
    budgets: tuple[int, ...]
    min_segment_samples: int
    shrinkage: float
    selective_min_samples: int
    selective_shrinkage: float
    source_devices: tuple[str, ...]
    target_devices: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budgets", default="8,16,32,64,128")
    parser.add_argument(
        "--target-devices",
        default="|".join(DEFAULT_TARGET_DEVICES),
        help="Pipe-separated exact Device values.",
    )
    parser.add_argument("--min-segment-samples", type=int, default=3)
    parser.add_argument("--shrinkage", type=float, default=4.0)
    parser.add_argument("--selective-min-samples", type=int, default=8)
    parser.add_argument("--selective-shrinkage", type=float, default=16.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_device_configs(data_root: Path) -> pd.DataFrame:
    records: dict[str, dict] = {}
    for split in ("train", "test"):
        for path in (data_root / split / "device_configs").glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            records[record["Device"]] = record
    if not records:
        raise FileNotFoundError(f"No device configs under {data_root}")
    return pd.DataFrame(records.values())


def parse_tile(kernel_name: str) -> tuple[float, float]:
    match = re.search(r"(?:sgemm_|tilesize)(\d+)x(\d+)", kernel_name.lower())
    if match is None:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def path_family(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "xmma" in name:
        return "tensor"
    if "cutlass" in name and "simt" in name:
        return "cutlass_simt"
    if "sliced" in name:
        return "sliced"
    return "library_sgemm"


def wave_bucket(waves: float) -> str:
    if waves <= 1:
        return "w1"
    if waves <= 4:
        return "w2_4"
    if waves <= 16:
        return "w5_16"
    if waves <= 64:
        return "w17_64"
    return "w65p"


def log_positive(series: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    return np.log(np.maximum(values, 1e-18))


def engineer_features(raw: pd.DataFrame, device_df: pd.DataFrame) -> pd.DataFrame:
    df = raw.merge(device_df, on="Device", how="left", validate="many_to_one")
    if df[["SingleFLOPs", "Mem_Bw", "Num_Sm"]].isna().any().any():
        missing = sorted(df.loc[df["SingleFLOPs"].isna(), "Device"].unique())
        raise ValueError(f"Missing device configs: {missing}")

    tiles = df["Kernel Name"].map(parse_tile)
    df["tile_1"] = [item[0] for item in tiles]
    df["tile_2"] = [item[1] for item in tiles]
    parse_failures = int(df["tile_1"].isna().sum())
    if parse_failures:
        raise ValueError(f"Unable to parse {parse_failures} kernel tile names")

    # NeuSight maps kernel tile_2 to M and tile_1 to K for its MM wave model.
    aligned_m = np.ceil(df["M"] / df["tile_2"]) * df["tile_2"]
    aligned_k = np.ceil(df["K"] / df["tile_1"]) * df["tile_1"]
    df["flops"] = 2.0 * df["B"] * aligned_m * df["N"] * aligned_k
    df["bytes"] = 4.0 * df["B"] * (
        df["M"] * df["N"] + df["N"] * df["K"] + df["M"] * df["K"]
    )
    df["arith_intensity"] = df["flops"] / df["bytes"].clip(lower=1.0)
    df["ridge"] = df["SingleFLOPs"] / df["Mem_Bw"]
    df["compute_floor_ms"] = df["flops"] / (df["SingleFLOPs"] * 1e9) * 1e3
    df["memory_floor_ms"] = df["bytes"] / (df["Mem_Bw"] * 1e9) * 1e3
    df["roofline_ms"] = df[["compute_floor_ms", "memory_floor_ms"]].max(axis=1)
    df["blocks"] = df["Grid x"] * df["Grid y"] * df["Grid z"]
    df["waves"] = np.ceil(df["blocks"] / df["Num_Sm"])
    df["regime"] = np.where(
        df["arith_intensity"] >= df["ridge"], "compute", "memory"
    )
    df["path_family"] = df["Kernel Name"].map(path_family)
    df["wave_bucket"] = df["waves"].map(wave_bucket)
    df["coarse_segment"] = df["regime"] + "|" + df["wave_bucket"]
    df["fine_segment"] = (
        df["path_family"] + "|" + df["regime"] + "|" + df["wave_bucket"]
    )

    log_sources = {
        "B": "B",
        "M": "M",
        "N": "N",
        "K": "K",
        "grid_x": "Grid x",
        "grid_y": "Grid y",
        "grid_z": "Grid z",
        "block_x": "Block x",
        "tile_1": "tile_1",
        "tile_2": "tile_2",
        "blocks": "blocks",
        "waves": "waves",
        "flops": "flops",
        "bytes": "bytes",
        "ai": "arith_intensity",
        "ridge": "ridge",
        "compute_floor_ms": "compute_floor_ms",
        "memory_floor_ms": "memory_floor_ms",
        "roofline_ms": "roofline_ms",
        "peak_gflops": "SingleFLOPs",
        "mem_bw": "Mem_Bw",
        "num_sm": "Num_Sm",
        "l2_cache": "L2Cache",
        "device_mem": "Dev_Mem",
    }
    for output_name, input_name in log_sources.items():
        df[f"log_{output_name}"] = log_positive(df[input_name])
    df["log_ai_over_ridge"] = log_positive(df["arith_intensity"] / df["ridge"])
    df["regime_compute"] = (df["regime"] == "compute").astype(float)
    for family in ("tensor", "cutlass_simt", "sliced", "library_sgemm"):
        df[f"path_{family}"] = (df["path_family"] == family).astype(float)
    return df


def fit_models(train: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    x = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    direct_target = log_positive(train["Latency"])
    slowdown = np.maximum(train["Latency"] / train["roofline_ms"], 1.0)
    residual_target = log_positive(slowdown)
    common = dict(
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        # Fixed source-only training avoids an internal random validation split
        # that can place the same shape from different source GPUs on both sides.
        early_stopping=False,
        random_state=20260810,
    )
    direct = HistGradientBoostingRegressor(**common).fit(x, direct_target)
    gray = HistGradientBoostingRegressor(**common).fit(x, residual_target)
    return direct, gray


def add_base_predictions(
    df: pd.DataFrame,
    direct: HistGradientBoostingRegressor,
    gray: HistGradientBoostingRegressor,
) -> pd.DataFrame:
    result = df.copy()
    x = result[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    result["pred_roofline"] = result["roofline_ms"]
    result["pred_direct"] = np.exp(direct.predict(x))
    result["pred_direct_clamped"] = np.maximum(
        result["pred_direct"], result["roofline_ms"]
    )
    log_slowdown = np.maximum(gray.predict(x), 0.0)
    result["pred_gray"] = result["roofline_ms"] * np.exp(log_slowdown)
    return result


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ape = np.abs(pred - y) / np.maximum(y, 1e-18) * 100.0
    return {
        "n_eval": float(len(y)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "p95_ape": float(np.quantile(ape, 0.95)),
        "wape": float(np.sum(np.abs(pred - y)) / np.sum(y) * 100.0),
        "within_10": float(np.mean(ape <= 10.0) * 100.0),
        "within_20": float(np.mean(ape <= 20.0) * 100.0),
        "underprediction_rate": float(np.mean(pred < y) * 100.0),
        "dangerous_under_50": float(np.mean(pred < 0.5 * y) * 100.0),
        "log_rmse": float(np.sqrt(np.mean((log_positive(pred) - log_positive(y)) ** 2))),
    }


def select_calibration_indices(
    target: pd.DataFrame, budget: int, seed: int, sampler: str
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_indices = target.index.to_numpy()
    budget = min(budget, len(all_indices) - 1)
    if sampler == "random":
        return np.sort(rng.choice(all_indices, size=budget, replace=False))
    if sampler != "coverage":
        raise ValueError(sampler)

    grouped: list[list[int]] = []
    for _, group in target.groupby("fine_segment", sort=True):
        values = group.index.to_numpy().copy()
        rng.shuffle(values)
        grouped.append(values.tolist())
    rng.shuffle(grouped)
    selected: list[int] = []
    while len(selected) < budget and any(grouped):
        remaining: list[list[int]] = []
        for group in grouped:
            if group and len(selected) < budget:
                selected.append(group.pop())
            if group:
                remaining.append(group)
        grouped = remaining
    return np.sort(np.asarray(selected, dtype=int))


def calibration_corrections(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    base_column: str,
    method: str,
    min_samples: int,
    shrinkage: float,
    selective_min_samples: int,
    selective_shrinkage: float,
) -> tuple[np.ndarray, dict[str, float]]:
    base_cal = np.maximum(calibration[base_column].to_numpy(), 1e-18)
    log_ratio = log_positive(calibration["Latency"].to_numpy() / base_cal)
    global_correction = float(np.median(log_ratio))
    if method == "global":
        correction = np.full(len(evaluation), global_correction)
        route_stats = {"fine_route": 0.0, "coarse_route": 0.0, "global_route": 100.0}
    elif method in ("segmented", "selective"):
        working = calibration[["fine_segment", "coarse_segment"]].copy()
        working["log_ratio"] = log_ratio
        maps: dict[str, dict[str, tuple[float, int, float]]] = {}
        for level in ("fine_segment", "coarse_segment"):
            level_map: dict[str, tuple[float, int, float]] = {}
            for key, group in working.groupby(level)["log_ratio"]:
                median = float(group.median())
                mad = float(np.median(np.abs(group.to_numpy() - median)))
                level_map[str(key)] = (median, int(len(group)), mad)
            maps[level] = level_map

        corrections: list[float] = []
        routes: list[str] = []
        for _, row in evaluation.iterrows():
            chosen = global_correction
            route = "global"
            for level, route_name in (
                ("fine_segment", "fine"),
                ("coarse_segment", "coarse"),
            ):
                candidate = maps[level].get(str(row[level]))
                if candidate is None:
                    continue
                median, count, mad = candidate
                if method == "segmented" and count >= min_samples:
                    weight = count / (count + shrinkage)
                    chosen = weight * median + (1.0 - weight) * global_correction
                    route = route_name
                    break
                if method == "selective" and count >= selective_min_samples:
                    delta = median - global_correction
                    robust_se = 1.4826 * mad / math.sqrt(count)
                    threshold = max(math.log(1.05), 1.96 * robust_se)
                    if abs(delta) > threshold:
                        weight = count / (count + selective_shrinkage)
                        chosen = global_correction + weight * delta
                        route = route_name
                        break
            corrections.append(chosen)
            routes.append(route)
        correction = np.asarray(corrections)
        route_stats = {
            "fine_route": float(np.mean(np.asarray(routes) == "fine") * 100.0),
            "coarse_route": float(np.mean(np.asarray(routes) == "coarse") * 100.0),
            "global_route": float(np.mean(np.asarray(routes) == "global") * 100.0),
        }
    else:
        raise ValueError(method)

    prediction = evaluation[base_column].to_numpy() * np.exp(correction)
    prediction = np.maximum(prediction, evaluation["roofline_ms"].to_numpy())
    return prediction, route_stats


def evaluate_zero_shot(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    model_columns = {
        "roofline": "pred_roofline",
        "direct": "pred_direct",
        "direct_clamped": "pred_direct_clamped",
        "gray_residual": "pred_gray",
    }
    for device, group in test.groupby("Device", sort=True):
        status = "source_gpu_unseen_shape" if device in SOURCE_DEVICES else "unseen_gpu_and_shape"
        for model_name, column in model_columns.items():
            row = {
                "device": device,
                "device_status": status,
                "model": model_name,
            }
            row.update(metrics(group["Latency"].to_numpy(), group[column].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_calibration(
    test: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    base_columns = {
        "direct_clamped": "pred_direct_clamped",
        "gray_residual": "pred_gray",
    }
    for device in config.target_devices:
        target = test[test["Device"] == device].copy().reset_index(drop=True)
        if target.empty:
            raise ValueError(f"No target rows for {device}")
        for sampler in ("random", "coverage"):
            for budget in config.budgets:
                for seed in range(config.seeds):
                    cal_indices = select_calibration_indices(target, budget, seed, sampler)
                    calibration = target.loc[cal_indices]
                    evaluation = target.drop(index=cal_indices)
                    for base_name, base_column in base_columns.items():
                        for method in ("global", "segmented", "selective"):
                            prediction, route_stats = calibration_corrections(
                                calibration,
                                evaluation,
                                base_column,
                                method,
                                config.min_segment_samples,
                                config.shrinkage,
                                config.selective_min_samples,
                                config.selective_shrinkage,
                            )
                            row = {
                                "device": device,
                                "sampler": sampler,
                                "budget": budget,
                                "seed": seed,
                                "base_model": base_name,
                                "calibration": method,
                                "fine_segments_sampled": calibration["fine_segment"].nunique(),
                                "fine_segments_total": target["fine_segment"].nunique(),
                            }
                            row.update(route_stats)
                            row.update(metrics(evaluation["Latency"].to_numpy(), prediction))
                            rows.append(row)
    return pd.DataFrame(rows)


def summarize_calibration(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mape",
        "median_ape",
        "p90_ape",
        "p95_ape",
        "wape",
        "within_10",
        "within_20",
        "underprediction_rate",
        "dangerous_under_50",
        "log_rmse",
        "fine_route",
        "coarse_route",
        "global_route",
        "fine_segments_sampled",
    ]
    grouped = raw.groupby(
        ["device", "sampler", "budget", "base_model", "calibration"], sort=True
    )[metric_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    return mean.join(std).reset_index()


def plot_results(zero: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    devices = list(DEFAULT_TARGET_DEVICES)
    fig, axes = plt.subplots(1, len(devices), figsize=(16, 4.6), sharey=False)
    if len(devices) == 1:
        axes = [axes]
    styles = {
        ("direct_clamped", "global"): ("--", "o", "Direct + global"),
        ("direct_clamped", "segmented"): ("-", "o", "Direct + segmented"),
        ("gray_residual", "global"): ("--", "s", "Gray + global"),
        ("gray_residual", "segmented"): ("-", "s", "Gray + segmented"),
        ("gray_residual", "selective"): ("-.", "^", "Gray + selective"),
    }
    for ax, device in zip(axes, devices):
        subset = summary[(summary["device"] == device) & (summary["sampler"] == "coverage")]
        for key, (linestyle, marker, label) in styles.items():
            series = subset[
                (subset["base_model"] == key[0]) & (subset["calibration"] == key[1])
            ].sort_values("budget")
            ax.plot(
                series["budget"],
                series["mape_mean"],
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        ax.set_title(device.replace("NVIDIA ", ""))
        ax.set_xlabel("Target measurements used")
        ax.set_ylabel("MAPE (%)")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "calibration_mape.png", dpi=180)
    plt.close(fig)

    pivot = zero[zero["device"].isin(devices)].pivot(
        index="device", columns="model", values="mape"
    )
    pivot = pivot[["roofline", "direct", "direct_clamped", "gray_residual"]]
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("Zero-shot MAPE (%)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "zero_shot_mape.png", dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    budgets = tuple(int(item) for item in args.budgets.split(",") if item)
    target_devices = tuple(item for item in args.target_devices.split("|") if item)
    config = ExperimentConfig(
        seeds=args.seeds,
        budgets=budgets,
        min_segment_samples=args.min_segment_samples,
        shrinkage=args.shrinkage,
        selective_min_samples=args.selective_min_samples,
        selective_shrinkage=args.selective_shrinkage,
        source_devices=tuple(SOURCE_DEVICES),
        target_devices=target_devices,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(args.data_root / "train" / "linear.csv")
    test_raw = pd.read_csv(args.data_root / "test" / "linear.csv")
    device_df = load_device_configs(args.data_root)
    train = engineer_features(train_raw, device_df)
    test = engineer_features(test_raw, device_df)
    train = train[train["Device"].isin(SOURCE_DEVICES)].copy()

    train_shapes = set(map(tuple, train[["B", "M", "N", "K"]].to_numpy()))
    target_test = test[test["Device"].isin(target_devices)]
    overlap = sum(
        tuple(row) in train_shapes
        for row in target_test[["B", "M", "N", "K"]].to_numpy()
    )
    if overlap:
        raise AssertionError(f"Shape leakage detected: {overlap} target rows")

    direct, gray = fit_models(train)
    train = add_base_predictions(train, direct, gray)
    test = add_base_predictions(test, direct, gray)
    zero = evaluate_zero_shot(test)
    raw_calibration = evaluate_calibration(test, config)
    summary = summarize_calibration(raw_calibration)

    zero.to_csv(args.output_dir / "zero_shot.csv", index=False)
    raw_calibration.to_csv(args.output_dir / "calibration_runs.csv", index=False)
    summary.to_csv(args.output_dir / "calibration_summary.csv", index=False)

    bound_violations = {
        "train_latency_below_roofline": int((train["Latency"] < train["roofline_ms"]).sum()),
        "test_latency_below_roofline": int((test["Latency"] < test["roofline_ms"]).sum()),
    }
    metadata = {
        "config": asdict(config),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "data": {
            "data_root": str(args.data_root.resolve()),
            "train_rows": len(train),
            "test_rows": len(test),
            "target_rows": {device: int((test["Device"] == device).sum()) for device in target_devices},
            "target_shape_overlap_with_train": overlap,
        },
        "features": FEATURE_COLUMNS,
        "roofline_bound_diagnostics": bound_violations,
        "assumption": "Kernel name/grid/tile are treated as compiler/profiler-visible inputs; algorithm selection is not predicted.",
        "scope": "Post-kernel component-cost experiment for FP32 Linear/GEMM only; not a pre-kernel tactic selector or end-to-end model.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not args.no_plots:
        plot_results(zero, summary, args.output_dir)

    print("Zero-shot target-device MAPE (%)")
    print(
        zero[zero["device"].isin(target_devices)]
        .pivot(index="device", columns="model", values="mape")
        .round(2)
        .to_string()
    )
    print("\nCoverage-sampled gray statistically selective calibration")
    selected = summary[
        (summary["sampler"] == "coverage")
        & (summary["base_model"] == "gray_residual")
        & (summary["calibration"] == "selective")
    ][["device", "budget", "mape_mean", "mape_std", "p95_ape_mean", "wape_mean"]]
    print(selected.round(2).to_string(index=False))
    print(f"\nResults: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
