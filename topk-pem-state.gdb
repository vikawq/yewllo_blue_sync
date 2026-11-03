# Read-only state decoder for the exact AArch64 PEM disassembly supplied in
# this chat. It reads registers/memory; it never calls target functions or
# changes target state. Code fingerprints guard the decoded structure offsets.
# The extended offsets below follow the candidate PEM layout supplied by the
# other investigation: 0x408=hold_id_, 0x40c=wait_id_, 0x438=pc_. The fingerprint
# checks are not a complete ABI/build-ID check; preserve raw offset names so
# these values can be checked against the deployed source/binary layout.
set pagination off
set print thread-events off
python
import collections
import gdb
import json

inferior = gdb.selected_inferior()

def read_uint(address, size):
    return int.from_bytes(bytes(inferior.read_memory(address, size)), "little")

def address_of(symbol):
    return int(gdb.parse_and_eval("&" + symbol))

def register(name):
    return int(gdb.parse_and_eval("$" + name))

driver = address_of("_Z22pem_drive_free_run_subP12pem_aic_coreP15aic_wrapper_ctxjj")
scheduler = address_of("_ZN18MultiCoreScheduler22wait_all_done_free_runEv")
driver_verified = (
    read_uint(driver + 1100, 4) == 0x97fa2070
    and read_uint(driver + 1104, 4) == 0x17ffff60
)
scheduler_verified = (
    read_uint(scheduler + 60, 4) == 0xf9402660
    and read_uint(scheduler + 68, 4) == 0xb9400661
)
print(json.dumps({
    "kind": "decoder",
    "pid": inferior.pid,
    "driver_code_matches": driver_verified,
    "scheduler_code_matches": scheduler_verified,
    "note": "Raw offset names retained. Candidate layout: 0x408=hold_id_, 0x40c=wait_id_, 0x438=pc_; 0x5a is the reported unset-ID sentinel. Validate deployed ABI. loop_w24 is host polling, not retired instructions.",
}))

counts = collections.Counter()
for thread in inferior.threads():
    row = {"kind": "thread", "gdb_thread": thread.num,
           "lwp": thread.ptid[1], "name": thread.name}
    try:
        thread.switch()
        frame = gdb.newest_frame()
        while frame is not None:
            name = frame.name() or ""
            if "pem_drive_free_run_sub" in name:
                frame.select()
                offset = frame.pc() - driver
                row.update(role="pem_driver", frame_offset=offset)
                counts["pem_driver_frames"] += 1
                if not driver_verified or offset != 1104:
                    row["decode_skipped"] = "Only the verified sched_yield return site can be decoded."
                else:
                    slot = register("x19")
                    core = read_uint(slot, 8)
                    row.update(
                        slot=hex(slot), core=hex(core),
                        progress_w21=register("x21") & 0xffffffff,
                        loop_w24=register("x24") & 0xffffffff,
                    )
                    if core:
                        row.update(
                            flag_0x430=read_uint(core + 0x430, 1),
                            flag_0x431=read_uint(core + 0x431, 1),
                            flag_0x432=read_uint(core + 0x432, 1),
                            u32_at_0x408=read_uint(core + 0x408, 4),
                            u32_at_0x40c=read_uint(core + 0x40c, 4),
                            u64_at_0x438=hex(read_uint(core + 0x438, 8)),
                            rvecpv_pointer_0x160=hex(read_uint(core + 0x160, 8)),
                            simdpv_pointer_0x170=hex(read_uint(core + 0x170, 8)),
                        )
                    counts["decoded_yield_sites"] += 1
                print(json.dumps(row))
                break
            if "wait_all_done_free_run" in name:
                frame.select()
                offset = frame.pc() - scheduler
                row.update(role="scheduler_wait", frame_offset=offset)
                if not scheduler_verified or offset != 60:
                    row["decode_skipped"] = "Only the verified step_all return site can be decoded."
                else:
                    obj = register("x19")
                    row.update(
                        scheduler_object=hex(obj),
                        state_u32_at_0x4=read_uint(obj + 4, 4),
                        target_u64_at_0x48=read_uint(obj + 72, 8),
                        limit_u64_at_0x38=read_uint(obj + 56, 8),
                        iteration_x20=register("x20"),
                        cached_limit_x21=register("x21"),
                    )
                counts["scheduler_wait_frames"] += 1
                print(json.dumps(row))
                break
            frame = frame.older()
    except (gdb.error, RuntimeError, ValueError) as error:
        row["error"] = str(error)
        print(json.dumps(row))
print(json.dumps({"kind": "summary", "counts": dict(counts)}))
end
detach
