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


def _bf16_rank(values):
    bits = values.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    magnitude = bits & 0x7FFF
    return torch.where(bits < 0, -magnitude, magnitude)


def _round_bf16_from_fp64(values):
    rounded = values.to(torch.bfloat16)
    candidate = rounded.double()
    rank = _bf16_rank(rounded)
    neighbor_rank = rank + torch.where(values > candidate, 1, -1)
    neighbor_bits = torch.where(neighbor_rank < 0, -0x8000 - neighbor_rank, neighbor_rank)
    neighbor = neighbor_bits.to(torch.int16).contiguous().view(torch.bfloat16).double()

    # Compare distances in FP64 in case the initial cast rounded through FP32.
    candidate_error = (values - candidate).abs()
    neighbor_error = (values - neighbor).abs()
    use_neighbor = (neighbor_error < candidate_error) | (
        (neighbor_error == candidate_error) & ((neighbor_rank & 1) == 0)
    )
    return torch.where(use_neighbor & torch.isfinite(candidate), neighbor, candidate)


def _probe_bf16_failures(group, mask, count, a, b, actual, ref):
    if count > 4096:
        print(f"[group-gemm] group={group} fp64_check=skipped mismatches={count}", flush=True)
        return

    actual_exact = ref_exact = neither_exact = one_ulp = nonfinite = 0
    with torch.autocast(device_type="cpu", enabled=False):
        for positions in mask.nonzero().split(64):
            rows, cols = positions.unbind(dim=1)
            actual_values = actual[rows, cols]
            ref_values = ref[rows, cols]
            lhs = a.index_select(0, rows).double()
            rhs = b.index_select(1, cols).t().double()
            fp64 = (lhs * rhs).sum(dim=1)
            rounded_fp64 = _round_bf16_from_fp64(fp64)
            actual_match = actual_values == rounded_fp64
            ref_match = ref_values == rounded_fp64
            finite = torch.isfinite(actual_values) & torch.isfinite(ref_values) & torch.isfinite(fp64)
            distance = (_bf16_rank(actual_values) - _bf16_rank(ref_values)).abs()
            actual_exact += int((actual_match & finite).sum())
            ref_exact += int((ref_match & finite).sum())
            neither_exact += int((~(actual_match | ref_match) & finite).sum())
            one_ulp += int(((distance == 1) & finite).sum())
            nonfinite += int((~finite).sum())

    print(
        f"[group-gemm] group={group} fp64_checked={count} "
        f"actual_exact_fp64={actual_exact} ref_exact_fp64={ref_exact} "
        f"neither_exact_fp64={neither_exact} one_bf16_ulp={one_ulp} nonfinite={nonfinite}",
        flush=True,
    )


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
    smallest_nonempty_group = min(
        (group for group, count in enumerate(row_counts) if count > 0),
        key=row_counts.__getitem__,
        default=None,
    )
    start = 0
    sampled = set()
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
        actual_mask = _mismatch(actual_group, expected, atol, rtol)
        baseline_count = int(baseline_mask.sum())
        actual_count = int(actual_mask.sum())
        ref_count = int(_mismatch(ref_group, expected, atol, rtol).sum())
        print(
            f"[group-gemm] group={group} rows=[{start},{stop}) "
            f"total={baseline_mask.numel()} actual_vs_ref={baseline_count} "
            f"actual_vs_cpu_fp32={actual_count} ref_vs_cpu_fp32={ref_count}",
            flush=True,
        )

        if baseline_count and actual.dtype == ref.dtype == x.dtype == torch.bfloat16:
            _probe_bf16_failures(group, baseline_mask, baseline_count, a, b, actual_group, ref_group)

        if group == smallest_nonempty_group and a.dtype == b.dtype == torch.bfloat16:
            try:
                with torch.autocast(device_type="cpu", enabled=False):
                    native_bf16 = torch.mm(a, b)
            except (RuntimeError, NotImplementedError) as error:
                print(f"[group-gemm] group={group} native_cpu_error={error}", flush=True)
            else:
                native = native_bf16.float()
                native_count = int(_mismatch(native, expected, atol, rtol).sum())
                native_ref_count = int(_mismatch(ref_group, native, atol, rtol).sum())
                print(
                    f"[group-gemm] group={group} native_cpu_dtype={native_bf16.dtype} "
                    f"native_cpu_vs_cpu_fp32={native_count} "
                    f"ref_vs_native_cpu={native_ref_count}",
                    flush=True,
                )

        for comparison, mask, count in (
            ("actual_vs_ref", baseline_mask, baseline_count),
            ("actual_vs_cpu_fp32", actual_mask, actual_count),
        ):
            if not count or comparison in sampled:
                continue
            row, col = _first_mismatch(mask)
            with torch.autocast(device_type="cpu", enabled=False):
                fp64 = torch.dot(a[row].double(), b[:, col].double())
                rounded_fp64 = (
                    _round_bf16_from_fp64(fp64.reshape(1))[0]
                    if x.dtype == torch.bfloat16
                    else fp64.to(x.dtype)
                )
            print(
                f"[group-gemm] sample={comparison} row={start + row} col={col} "
                f"actual={actual_group[row, col].item()} ref={ref_group[row, col].item()} "
                f"cpu_fp32={fp32[row, col].item()} cpu_fp64={fp64.item()} "
                f"rounded_fp64={rounded_fp64.item()}",
                flush=True,
            )
            sampled.add(comparison)

        start = stop
        del fp32, expected, actual_group, ref_group, baseline_mask, actual_mask, mask

    check_tol_diff(actual, ref, atol, rtol)
