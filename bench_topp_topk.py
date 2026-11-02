"""Standalone NPU TopK diagnosis with optional NPU-Sim launch evidence.

Run each shape/extra-memory configuration in a fresh process. Times are host
wall-clock measurements, including simulation when a simulator is active.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch_npu  # noqa: F401


def _launches(session):
    if session is None:
        return set()
    return {
        path for pattern in ("hal-launch-*", "launch-*")
        for path in session.glob(pattern) if path.is_dir()
    }


def _launch_evidence(directory):
    row = {"directory": str(directory)}
    try:
        mappings = directory / "mappings.tsv"
        if mappings.is_file():
            sizes = [int(line.split("\t")[1]) for line in mappings.read_text().splitlines() if line.strip()]
            row.update(mapped_allocations=len(sizes), mapped_bytes=sum(sizes))
        result_file = directory / "result.json"
        if result_file.is_file():
            result = json.loads(result_file.read_text())
            row.update({key: result[key] for key in (
                "backend_id", "engine_id", "launch_id", "instructions", "programs",
                "host_workers", "host_vector_backend", "advance_calls", "pipeline_operations",
            ) if key in result})
        row["trace_bytes"] = (directory / "trace.jsonl").stat().st_size if (directory / "trace.jsonl").is_file() else 0
    except (OSError, ValueError, IndexError) as error:
        row["evidence_error"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--extra-mib", type=int, default=0,
                        help="Keep this much initialized, unused NPU memory alive during TopK.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()
    if args.batch < 1 or args.vocab < 1 or not 1 <= args.k <= args.vocab or args.repeat < 1 or args.extra_mib < 0:
        parser.error("Invalid shape, k, repeat or extra-mib.")

    names = (
        "NPUSIM_INTERCEPT_LAYER", "NPUSIM_SESSION_DIR", "NPUSIM_HAL_DIRECT",
        "NPUSIM_DEVICE_BACKEND_LIBRARY", "NPUSIM_BACKEND", "NPUSIM_WORKERS",
        "NPUSIM_TRACE", "NPUSIM_HAL_CAPTURE", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING",
    )
    environment = {name: os.environ.get(name) for name in names}
    report = {
        "pid": os.getpid(), "torch": str(torch.__version__),
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
        "arguments": vars(args), "environment": environment, "calls": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "calls"}, indent=2), flush=True)
    session = Path(environment["NPUSIM_SESSION_DIR"]) if environment["NPUSIM_SESSION_DIR"] else None
    if session is None:
        print("No NPUSIM_SESSION_DIR: launch evidence unavailable; timings still collected.", flush=True)

    print("Preparing FP32 input and optional extra memory (excluded from timing)...", flush=True)
    torch.manual_seed(20260806)
    x = torch.randn((args.batch, args.vocab), dtype=torch.float32, device="cpu").npu()
    extra = None
    if args.extra_mib:
        extra = torch.zeros(args.extra_mib * 1024 * 1024, dtype=torch.uint8, device="cpu").npu()
    torch.npu.synchronize()
    print("Input ready.", flush=True)

    with torch.inference_mode():
        for iteration in range(1, args.repeat + 1):
            before = _launches(session)
            print(f"[topk {iteration}/{args.repeat}] BEGIN", flush=True)
            start = time.perf_counter()
            values, indices = torch.topk(x, args.k, dim=-1, largest=True, sorted=True)
            host_ms = (time.perf_counter() - start) * 1000
            print(f"[topk {iteration}] HOST_RETURN={host_ms:.3f} ms; SYNC_BEGIN", flush=True)
            start = time.perf_counter()
            torch.npu.synchronize()
            sync_ms = (time.perf_counter() - start) * 1000
            new = sorted(_launches(session) - before, key=lambda path: int(path.name.rsplit("-", 1)[1]))
            launches = [_launch_evidence(path) for path in new]
            call = {"iteration": iteration, "host_ms": host_ms, "sync_ms": sync_ms,
                    "total_ms": host_ms + sync_ms,
                    "launch_count": len(launches) if session is not None else None, "launches": launches}
            report["calls"].append(call)
            print(json.dumps(call, indent=2), flush=True)
            if args.output:
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del values, indices
    # Keep the unused allocation alive until every measurement is complete.
    del extra
    if args.output:
        print(f"Report: {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
