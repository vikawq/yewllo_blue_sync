import torch


def bf16_safe_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run the BF16 CPU offload path with FP32 inputs and a BF16 result."""
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("bf16_safe_mm expects both operands on CPU")

    if a.dtype == b.dtype == torch.bfloat16:
        with torch.autocast(device_type="cpu", enabled=False):
            return torch.ops.aten.mm.default(a.float(), b.float()).to(a.dtype)

    return torch.ops.aten.mm.default(a, b)


def bf16_safe_2d_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply the MM exception if the frontend intercepts matmul before decomposition."""
    if a.ndim == b.ndim == 2:
        return bf16_safe_mm(a, b)

    return torch.ops.aten.matmul.default(a, b)
