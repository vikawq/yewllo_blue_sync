"""直接读 HAL dump 下来的参数块, 把 grid 到底是几看清楚 —— 不再靠推断。

发现: NPU-Sim 的 code/backend/ascend950/loader/image_bridge.inc:94 每次 launch 都会把
    kernel.o / args.bin / mappings.tsv 落盘, 再调 npusim::launch_backend(..., trace.jsonl, ...)
也就是说**传给 kernel 的那块参数内存已经在磁盘上了**。而 hal_policy 算出的 grid_offset
(SWA 是 144) 就是 grid 在这块内存里的字节偏移。于是:

    args.bin[grid_offset : grid_offset+12] 解成 3 个 i32 = gridX, gridY, gridZ

SWA 报 `scalar_rem div by 0` 对应 IR 里的 `remsi block_idx, %arg24`, 而 %arg24 = gridX。
所以这一个读数就能定论:
    gridX == 0   -> grid 真的没传进参数块, 问题在 HAL 写 grid 这一环
    gridX == 预期 -> grid 是对的, div by 0 来自别处 (得去 trace.jsonl 找真正的除法指令)

本脚本只读文件, 不碰 NPU、不起仿真器。

用法 (cwd 留在 NPU-Sim):
    python _t_sim_dump_args.py                    # 自动找最新的 launch 目录
    python _t_sim_dump_args.py <某个目录>          # 指定目录
    ARGS_OFFSET=144 python _t_sim_dump_args.py    # 手动指定 grid_offset (默认从同级 metadata.json 取)
    ARGS_N=3 python _t_sim_dump_args.py           # 看最近 3 个 (默认 1)

顺带会按 _swa_fwd_kernel 的已知布局把整块参数解一遍 (9 个指针 + 13 个 4 字节 + 3 个 grid),
指针值能和 mappings.tsv 里的设备地址对一下, 偏错位了一眼就能看出来。
"""
import glob
import json
import os
import struct
import sys

# _swa_fwd_kernel 编译后的实际参数表 (27 个 arg, 见 MLIR 签名); 前两个是隐藏指针
SWA_LAYOUT = (
    [("sync_block_lock", "ptr"), ("workspace", "ptr")]
    + [(n, "ptr") for n in ("o", "o_f32", "lse", "q", "k", "v")]
    + [("bsz", "i32")]
    + [(n, "ptr") for n in ("cu_q_lens", "cu_total_seq_lens")]
    + [("scale", "f32")]
    + [(f"stride_{i}", "i32") for i in range(11)]
    + [("causal_mask", "ptr")]
    + [("gridX", "i32"), ("gridY", "i32"), ("gridZ", "i32")]
)


def find_dirs(limit):
    roots = []
    if os.environ.get("NPUSIM_SESSION_DIR"):
        roots.append(os.environ["NPUSIM_SESSION_DIR"])
    roots += [os.path.join(os.getcwd(), "build", "sessions"),
              os.path.join(os.getcwd(), "build")]
    found = set()
    for root in roots:
        found.update(os.path.dirname(p) for p in
                     glob.glob(os.path.join(root, "**", "args.bin"), recursive=True))
    return sorted(found, key=os.path.getmtime, reverse=True)[:limit]


def grid_offset_for(directory):
    if os.environ.get("ARGS_OFFSET"):
        return int(os.environ["ARGS_OFFSET"]), "ARGS_OFFSET"
    for cand in (os.path.join(directory, "metadata.json"),
                 os.path.join(os.path.dirname(directory), "metadata.json")):
        if os.path.isfile(cand):
            try:
                data = json.loads(open(cand, encoding="utf-8").read())
            except (OSError, ValueError):
                continue
            if data.get("grid_offset") is not None:
                return int(data["grid_offset"]), f"{cand} 的 kernel={data.get('kernel')}"
    return None, "没找到 metadata.json, 用 ARGS_OFFSET=N 指定"


def hexdump(blob, start=0, length=None):
    end = len(blob) if length is None else min(len(blob), start + length)
    for off in range(start, end, 16):
        chunk = blob[off:off + 16]
        words = " ".join(f"{struct.unpack_from('<I', chunk, i)[0]:08x}"
                         for i in range(0, len(chunk) - 3, 4))
        print(f"   {off:04d} (0x{off:03x}) | {words}", flush=True)


def report(directory):
    args_path = os.path.join(directory, "args.bin")
    blob = open(args_path, "rb").read()
    print(f"\n=== {directory}")
    for name in ("kernel.o", "mappings.tsv", "trace.jsonl", "ascendc-launch.bin",
                 "grid-offset.bin", "sqe.bin"):
        path = os.path.join(directory, name)
        print(f"   {name:<20} {os.path.getsize(path) if os.path.isfile(path) else '-':>10}",
              flush=True)
    print(f"   args.bin 共 {len(blob)} 字节")

    offset, how = grid_offset_for(directory)
    print(f"   grid_offset = {offset}  ({how})")
    print("   hexdump (小端 u32):")
    hexdump(blob)

    if offset is not None and offset + 12 <= len(blob):
        gx, gy, gz = struct.unpack_from("<3i", blob, offset)
        print(f"\n   >>> grid = ({gx}, {gy}, {gz})   乘积 = {gx * gy * gz}")
        if gx == 0:
            print("   >>> gridX 是 0 —— `remsi block_idx, gridX` 必然 div by 0,"
                  " grid 没进参数块", flush=True)
        else:
            print(f"   >>> gridX = {gx} 非 0 —— grid 进来了, div by 0 来自别处,"
                  " 去 trace.jsonl 找真正的除法指令", flush=True)
    elif offset is not None:
        print(f"\n   >>> args.bin 只有 {len(blob)} 字节, 放不下 offset={offset} 处的 grid"
              " —— 这本身就说明偏移和实际参数块不匹配", flush=True)

    # 按已知布局解一遍, 便于肉眼发现错位
    if len(blob) >= 180:
        print("\n   按 _swa_fwd_kernel 已知布局解 (对不上就说明不是这个 kernel):")
        pos = 0
        for name, kind in SWA_LAYOUT:
            size = 8 if kind == "ptr" else 4
            align = 8 if size == 8 else 4
            pos = (pos + align - 1) // align * align
            if pos + size > len(blob):
                break
            if kind == "ptr":
                (value,) = struct.unpack_from("<Q", blob, pos)
                shown = f"0x{value:012x}"
            elif kind == "f32":
                (value,) = struct.unpack_from("<f", blob, pos)
                shown = f"{value:g}"
            else:
                (value,) = struct.unpack_from("<i", blob, pos)
                shown = str(value)
            mark = "   <== grid" if name.startswith("grid") else ""
            print(f"     {pos:>4} {name:<18} {kind:<4} = {shown}{mark}", flush=True)
            pos += size

    mappings = os.path.join(directory, "mappings.tsv")
    if os.path.isfile(mappings):
        lines = open(mappings, encoding="utf-8", errors="replace").read().splitlines()
        print(f"\n   mappings.tsv ({len(lines)} 行, 设备地址 / 长度 / 输入 / 输出):")
        for line in lines[:12]:
            print(f"     {line}", flush=True)

    trace = os.path.join(directory, "trace.jsonl")
    if os.path.isfile(trace) and os.path.getsize(trace):
        lines = open(trace, encoding="utf-8", errors="replace").read().splitlines()
        print(f"\n   trace.jsonl 共 {len(lines)} 行, 头 3 行:")
        for line in lines[:3]:
            print(f"     {line[:200]}", flush=True)


def main():
    explicit = [a for a in sys.argv[1:] if os.path.isdir(a)]
    dirs = explicit or find_dirs(int(os.environ.get("ARGS_N", "1")))
    if not dirs:
        print("[args] 没找到任何 args.bin。", flush=True)
        print(f"[args] 找过 NPUSIM_SESSION_DIR={os.environ.get('NPUSIM_SESSION_DIR')!r}"
              f" 和 {os.path.join(os.getcwd(), 'build/sessions')}", flush=True)
        print("[args] 可以手动找: find <NPU-Sim>/build -name args.bin -newermt '-1 hour'",
              flush=True)
        return
    for directory in dirs:
        report(directory)


if __name__ == "__main__":
    main()
