"""SWA fwd 非法地址判决实验: 把输入张量填成常数, 看被拒地址是否跟着变成该常数的 bit pattern。

背景: test_swa_fwd_bwd_950_perf_triton.py::test_swa_function[True-4-255-M_F32] 在仿真上报
    [standalone_mem_bus] unmapped read/write @ 0x...  rejected
    [ERROR] scalar_func.h:1880 memory_map_calc_op0 Scalar: invalid addr.
被拒地址的低 32 位按 fp32 解码 44/44 都是 randn 量级的值 -> 怀疑 kernel 把张量数据当成了 GM 地址。

只跑第一个 Triton kernel (_swa_fwd_kernel), shape/参数与该用例完全一致:
    B=2 Q_H=16 KV_H=4 D=128 Q_LEN=1024 fp32, gqa_interleave=True, global=4, local=255, output_f32=True

用法 (cwd 留在 NPU-Sim 目录, 与跑 pytest 时相同; 脚本放进 mojo_opset 仓库任意子目录即可,
会自动把仓库根加进 sys.path; 下面的 python _t_swa_fill_probe.py 换成脚本的实际路径):
    python _t_swa_fill_probe.py                  # 不填, 复现原始报错 (基线)
    SWA_FILL=q    python _t_swa_fill_probe.py    # 只把 query 填成 -1.0
    SWA_FILL=k    python _t_swa_fill_probe.py
    SWA_FILL=v    python _t_swa_fill_probe.py
    SWA_FILL=all  python _t_swa_fill_probe.py
    SWA_FILL=q SWA_FILL_VALUE=0.0 python _t_swa_fill_probe.py

判读 (默认填 -1.0, fp32 bit pattern 0xBF800000):
  * 被填的那个张量就是"地址来源"时: 被拒地址收敛成 0xbf800000 附近 (8 字节读两个 -1.0,
    高 16 位被 48 位掩码截掉), 不再是一堆随机值
  * 填了却没变化: 地址不是从这个张量读的, 换下一个
  * 填 0.0 时 (0x00000000 落在仿真窗口 [0, 0x80000000) 内): unmapped 报错应基本消失,
    但数值必然错 —— 这一档只看有没有 unmapped, 不看精度
  * 0x5a5a5a5a5a5a 那几条是没写过的槽位, 预期填什么都不变
"""
import math
import os
import struct
import sys


def _add_repo_root_to_path():
    """脚本放在仓库任意子目录都能 import mojo_opset; cwd 仍须留在 NPU-Sim (仿真器按 cwd 找 etc/*.toml)。"""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "mojo_opset", "__init__.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


_add_repo_root_to_path()

import torch
import torch_npu  # noqa: F401

from mojo_opset.utils.platform import get_torch_device
from mojo_opset.backends.ttx.kernels.npu.swa import swa_fwd_impl

B, Q_H, KV_H, D, Q_LEN = 2, 16, 4, 128, 1024
DTYPE = torch.float32
GQA_INTERLEAVE, GLOBAL_WINDOW, LOCAL_WINDOW = True, 4, 255


def main():
    fill = os.environ.get("SWA_FILL", "").strip().lower()
    fill_value = float(os.environ.get("SWA_FILL_VALUE", "-1.0"))
    if fill not in ("", "q", "k", "v", "all"):
        sys.exit(f"SWA_FILL 只能是 q/k/v/all, 收到 {fill!r}")

    device = get_torch_device()
    torch.set_default_device(device)

    # 与 generate_sdpa_data 一致: 固定 seed, 等长 q_lens, 无 kv cache
    torch.manual_seed(43)
    q_lens = torch.tensor([Q_LEN] * B, dtype=torch.int32, device="cpu")
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32, device="cpu"),
                           torch.cumsum(q_lens, 0).to(torch.int32)])
    cu_total_seq_lens = cu_q_lens.clone()
    total = cu_q_lens[-1].item()

    query = torch.randn(total, Q_H, D, dtype=DTYPE, device="cpu")
    key = torch.randn(total, KV_H, D, dtype=DTYPE, device="cpu")
    value = torch.randn(total, KV_H, D, dtype=DTYPE, device="cpu")

    targets = {"q": [query], "k": [key], "v": [value], "all": [query, key, value]}.get(fill, [])
    for t in targets:
        t.fill_(fill_value)
    if fill:
        bits = struct.unpack("<I", struct.pack("<f", fill_value))[0]
        print(f"[probe] {fill} 填成 {fill_value!r} -> fp32 bits 0x{bits:08X}, "
              f"在仿真窗口内={bits < 0x80000000}")
    else:
        print("[probe] 不填, 基线")

    query, key, value = (t.to(device) for t in (query, key, value))
    cu_q_lens, cu_total_seq_lens = cu_q_lens.to(device), cu_total_seq_lens.to(device)

    swa_fwd_impl(
        q=query,
        k=key,
        v=value,
        cu_q_lens=cu_q_lens,
        cu_total_seq_lens=cu_total_seq_lens,
        is_causal=True,
        local_window_size=LOCAL_WINDOW,
        global_window_size=GLOBAL_WINDOW,
        softmax_scale=1.0 / math.sqrt(D),
        gqa_interleave=GQA_INTERLEAVE,
        output_f32=True,
    )
    torch.npu.synchronize()
    print("[probe] swa_fwd_impl 返回 (数值不校验, 只看上面的 unmapped / invalid addr 日志)")


if __name__ == "__main__":
    main()
