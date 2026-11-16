import torch


def _mismatch(actual, expected, atol, rtol):
    return ~torch.isclose(actual, expected, atol=atol, rtol=rtol)


def _first_mismatch(mask):
    rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return None
    row = int(rows[0])
    col = int(mask[row].nonzero(as_tuple=True)[0][0])
    return row, col


def probe_group_gemm_cpu(op, reference, x, group_list, *, atol, rtol):
    """Print a per-group CPU A/B comparison, then run the original assertion."""
    from mojo_opset.utils.acc import check_tol_diff

    print("[group-gemm] phase=actual", flush=True)
    actual = op.forward(x.clone(), group_list.clone())
    print("[group-gemm] phase=torch_reference", flush=True)
    ref = reference.forward(x.clone(), group_list.clone())
    print("[group-gemm] phase=cpu_fp32_probe", flush=True)
    print(
        f"[group-gemm] backend={type(op).__name__} x={x.dtype}/{x.device} "
        f"actual={actual.dtype}/{actual.device} ref={ref.dtype}/{ref.device} "
        f"trans_weight={reference.trans_weight}",
        flush=True,
    )
    if actual.shape != ref.shape:
        check_tol_diff(actual, ref, atol, rtol)

    row_counts = group_list.detach().cpu().tolist()
    if sum(row_counts) != x.shape[0] or min(row_counts, default=0) < 0:
        raise ValueError(f"Invalid Group GEMM row counts: {row_counts}")

    x_cpu = x.detach().cpu()
    actual_cpu = actual.detach().cpu()
    ref_cpu = ref.detach().cpu()
    start = 0
    sampled = False
    for group, count in enumerate(row_counts):
        if count == 0:
            continue
        stop = start + count
        a = x_cpu[start:stop]
        b = reference.weight[group].detach().cpu()
        if reference.trans_weight:
            b = b.t().contiguous()

        with torch.autocast(device_type="cpu", enabled=False):
            fp32 = torch.mm(a.float(), b.float())
        if fp32.dtype != torch.float32:
            raise RuntimeError(f"Expected FP32 CPU mm output, got {fp32.dtype}")

        expected = fp32.to(x.dtype).float()
        actual_group = actual_cpu[start:stop].float()
        ref_group = ref_cpu[start:stop].float()
        baseline_mask = _mismatch(actual_group, ref_group, atol, rtol)
        baseline_count = int(baseline_mask.sum())
        actual_count = int(_mismatch(actual_group, expected, atol, rtol).sum())
        ref_count = int(_mismatch(ref_group, expected, atol, rtol).sum())
        print(
            f"[group-gemm] group={group} rows=[{start},{stop}) "
            f"total={baseline_mask.numel()} actual_vs_ref={baseline_count} "
            f"actual_vs_cpu_fp32={actual_count} ref_vs_cpu_fp32={ref_count}",
            flush=True,
        )

        if baseline_count and not sampled:
            row, col = _first_mismatch(baseline_mask)
            with torch.autocast(device_type="cpu", enabled=False):
                fp64 = torch.dot(a[row].double(), b[:, col].double())
            print(
                f"[group-gemm] row={start + row} col={col} "
                f"actual={actual_group[row, col].item()} ref={ref_group[row, col].item()} "
                f"cpu_fp32={fp32[row, col].item()} cpu_fp64={fp64.item()} "
                f"rounded_fp64={fp64.to(x.dtype).item()}",
                flush=True,
            )
            sampled = True

        start = stop
        del fp32, expected, actual_group, ref_group, baseline_mask

    check_tol_diff(actual, ref, atol, rtol)
