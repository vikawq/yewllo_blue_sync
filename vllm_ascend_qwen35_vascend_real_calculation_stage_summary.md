# Qwen3.5-35B-A3B 接入 vAscend 真实计算阶段总结与 bug-fix 交接

> 日期：2026-08-03
> 模型：`/data/xuy/models/Qwen3.5-35B-A3B/`
> vLLM：`0.22.1`
> 服务名：`qwen3.5`
> 当前阶段：真实 NPU runner 已完成一次 `max_tokens=1` 的首 token 端到端请求；
> HTTP 200、`finish_reason="length"` 和 `[DONE]` 均已返回，EngineCore 正常收尾；
> 当前可见文本为空，token ID、数值正确性和输出质量仍待验证。

## 1. 文档范围

本文记录近一个月内，为把 Qwen3.5-35B-A3B 从本地 simulator/fallback 路径推进到
vAscend gRPC 真实 NPU 计算路径所做的关键修改、验证证据和遗留问题。

本次完整性审计以以下历史记录为基线：

```text
D:\workspaces\vLLM-ascend_for_lingqu\vllm_ascend_qwen35_debug_patches.md
D:\workspaces\vLLM-ascend_for_lingqu\vllm_ascend_qwen35_success_startup_changes.md
D:\workspaces\vLLM-ascend_for_lingqu\vllm_ascend_qwen35_startup_to_inference_success_changes.md
```

上述文档保留阶段过程和早期命令，本文件作为当前统一交接入口；出现状态冲突时，以本文件
标注的“当前状态”和实际工作区 diff 为准。

此前文档中记录过 simulator-only 模式下接口能够返回，但输出内容和数值并不可信。
本阶段的目标不同：

1. vLLM 在客户端容器正常启动。
2. ACLNN GetWorkspace 和执行函数被 `libcust_opapi.so` 正确截获。
3. 需要真实计算的算子通过 gRPC 发送给 NPU runner。
4. runner 使用真实 CANN `libopapi.so` 执行算子并返回 tensor。
5. 模型 forward 能持续越过已经发现的阻塞算子。

本文中的“首 token 跑通”指首轮 prefill、真实 NPU 算子执行、sampling 和 API 流式收尾
已经完整执行，不等同于“模型精度最终验收通过”。当前已经获得 HTTP 200 和完整的流式
结束标志，但最终验收仍要求确认原始 token ID、得到非空且合理的模型输出，并完成基础
数值一致性验证。

## 2. 当前结论

### 2.1 已达到的状态

- vLLM API Server 和 EngineCore 可以正常启动。
- text-only 请求已进入模型首轮 prefill。
- `aclnnUniqueConsecutive` 动态输出 shape 问题已解决，独立测试通过。
- GDN/causal-conv1d 原始阻塞点已被越过。
- `aclnnScatterPaKvCache` 已从客户端成功发送到真实 NPU runner。
- `aclnnFusedInferAttentionScoreV5` 已加入客户端生成代码、runner 注册和真实计算配置。
- 最新运行没有出现新的 CANN、ABI、segmentation fault 或 `std::bad_alloc` 错误。
- `max_tokens=1` 的请求已完成 prefill、sampling 和响应收尾。
- API 返回 HTTP 200、`finish_reason="length"` 和 `[DONE]`，EngineCore 没有 fatal error。
- 服务端记录 `Running: 0 reqs, Waiting: 0 reqs`，请求正常离开调度器。
- 当前阶段可以正式表述为“真实 NPU 首 token 生成链路已跑通”。

### 2.2 尚未达到的状态

- 首 token 生成步已完成，但流式响应中的可见 `content` 仍为空。
- 尚未确认生成的 token ID；不能排除特殊 token 被 `skip_special_tokens` 过滤。
- 尚未完成真实权重输出质量和数值正确性验收。
- 尚未完成性能优化；当前 no-stream、同步日志和逐算子 gRPC 路径非常慢。
- 日志中仍存在若干 `Output count mismatch` 警告，尚未逐个归因和清理。

## 3. 系统拓扑

当前链路由三个环境组成：

| 环境 | 主要职责 | 关键路径 |
| --- | --- | --- |
| vLLM 客户端容器 | 加载模型、调度、调用 torch_npu/ACLNN | `/data/xuy/qwen` |
| vAscend 编译容器 | 生成并编译 mock op API、runner | `/data/xuy/workspaces/vAscend` |
| 真实 NPU runner 机器 | 接收 gRPC 请求并通过真实 CANN 执行 | `141.61.41.197:50043` |

客户端关键环境变量：

```bash
SIMULATOR_HOME=/root/vascend_nostream
ASCEND_CUSTOM_OPP_PATH=/root/simulator/custom_op
VASCEND=1
VASCEND_GRPC_SOCKET_PATH=141.61.41.197:50043
ASCEND_LAUNCH_BLOCKING=1
```

当前主要运行模式：

```text
device=npu
enforce_eager=True
tensor_parallel_size=1
max_seq_len=2048
text-only
no-stream vAscend execution
```

## 4. 阶段推进记录

| 阶段 | 主要现象 | 处理结果 |
| --- | --- | --- |
| 服务启动 | 模型或插件初始化失败 | 已能稳定启动 API Server |
| simulator 推理 | 接口曾返回但内容异常 | 仅作为链路验证，不作为真实计算结果 |
| GDN | Triton、causal conv1d、state dtype 等错误 | 增加受控 PyTorch fallback，模型继续推进 |
| Unique | 返回数据正确但输出仍是标量 shape | 修复动态输出注册和执行顺序，独立测试通过 |
| Conv1D | `ge.aicoreNum=64` 超过模拟范围 `[0,4]` | 绕过本地 `F.conv1d` 编译路径，模型继续推进 |
| Scatter | `std::bad_alloc`，随后出现 SIGSEGV | 通过 GDB 定位 GetWorkspace ABI 参数错位 |
| Scatter 格式 | runner 报 PA_NZ cache 最后一维不合法 | vAscend RPC 将 `PA_NZ` 转换为 `Norm` |
| Attention V5 | 本地 CANN 找不到 tiling key | 补齐 V5 生成、runner 注册、配置和 SO 部署 |
| 首 token | `max_tokens=1` 完成真实计算、sampling 和流式收尾 | HTTP 200、`finish_reason="length"`、`[DONE]` |
| 当前 | 首 token 可见文本为空 | 主要问题转为 token 识别、输出正确性、告警清理和性能 |

## 5. 完整 bug-fix 覆盖审计

本节是交接索引。后续章节保留关键实现的详细根因和验证过程；本节负责回答“在哪个
地址、修改了什么、当前是否仍是有效方案”。审计来源包括三份早期排查文档、当前
Windows 工作区实际 diff、vAscend 生成代码、runner 注册和本轮端到端日志。

### 5.1 路径口径

项目推进期间 vLLM Ascend 源码在 Linux 容器中出现过两个根目录：

```text
早期运行目录：/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu
当前运行目录：/data/xuy/qwen/vLLM-ascend_for_lingqu
当前 upstream vLLM：/data/xuy/qwen/vllm
vAscend 编译目录：/data/xuy/workspaces/vAscend
模型目录：/data/xuy/models/Qwen3.5-35B-A3B
```

下面使用当前目录描述仍在使用的修改；仅存在于早期运行目录或临时容器的 patch 会明确
标记。Windows 交付工作区为：

```text
D:\workspaces\vLLM-ascend_for_lingqu
```

### 5.2 启动阶段 bug-fix

| ID | 原始问题 | 修改地址和符号 | 修改内容 | 当前状态 |
| --- | --- | --- | --- | --- |
| S01 | Qwen3VL image preprocess 的 `resample` / `interpolation` 签名不兼容 | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/patch/worker/patch_qwen3vl_image_preprocess.py`，`_patched_preprocess_compat` | 将 `resample` 转换为 patch 期望的 `interpolation` 后调用原实现 | 早期运行环境有效；当前 text-only 验证未覆盖多模态 |
| S02 | `vllm.triton_utils.triton` 缺少 `next_power_of_2` | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/triton/layernorm_gated.py` | 改用 `vllm.utils.math_utils.next_power_of_2`，或补同名 helper | 只能修 helper；不能修复 kernel launch，属于前置兼容修复 |
| S03 | gated LayerNorm 的 Triton kernel 是普通 Python function，`kernel[grid]` 报错 | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/layernorm.py`，`_rms_norm_gated_torch_fallback`、`forward_oot` | 使用 float32 中间计算实现 RMSNorm、SiLU gate、group norm，再转回原 dtype | 早期有效 fallback；性能和精度仍需正式回归 |
| S04 | 单卡场景 `expert_map=None`，直接 `.npu()` 崩溃 | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/fused_moe/token_dispatcher.py` | 仅在 `expert_map is not None` 时执行 `.npu()` | 有效兼容修复 |
| S05 | MoE token unpermute 路径使用 `torch_npu` 但未导入 | 同一 `token_dispatcher.py` 文件顶部 | 增加 `import torch_npu` | 有效缺失 import 修复 |
| S06 | `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope` 未注册 | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/patch/worker/patch_qwen3_5.py` | 显式 import `vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope` 触发注册 | 注册问题被解决，但 fused kernel 后续确认不可 launch |
| S07 | vectorcore 属性未初始化 | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py` | 尝试初始化 device properties，并曾以 `VLLM_ASCEND_VECTORCORE_FALLBACK=1` 兜底 | 仅诊断性绕过；kernel 仍不可 launch，不是最终方案 |
| S08 | Qwen3.5 fused attention monkey patch 强制进入不可用 Triton kernel | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/patch/worker/patch_qwen3_5.py`，`Qwen3NextAttention.forward` 赋值 | 注释 `Qwen3NextAttention.forward = AscendQwen3NextAttention.forward`，保留 GDN 相关 patch | 当前 eager/fallback 路径的关键绕行；恢复前必须验证 Triton/custom kernel |
| S09 | `_C_ascend::npu_causal_conv1d_custom` schema / PrivateUse1 kernel 未注册 | `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/csrc/torch_binding.cpp` | 将 `ops.def` 和 `ops.impl` 移到 `VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` 条件外 | C++ 注册已补；底层 `aclnnCausalConv1d` symbol 缺失问题由后续 GDN fallback 处理 |
| S10 | vLLM 版本间日志 formatter import 路径变化 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/logger.py`，`ColoredFormatter`、`NewLineFormatter` | 依次尝试 `vllm.logging_utils`、`vllm.logger`，最后提供本地 formatter 兜底 | 当前 Windows 工作区有实际 diff |
| S11 | mock `LD_PRELOAD` 影响 git、find、编译和依赖探测 | vLLM 容器运行环境，曾加载 `/usr/local/lib/libvnnopbase.so:/usr/local/lib/libhccl_mock.so:/usr/local/lib/libdlopen_vllm.so` | 服务运行保留所需 preload；构建和检查命令使用 `env -u LD_PRELOAD ...` | 有效操作规范，不是源码修改 |
| S12 | graph/Triton和多模态路径会引入尚未解决的变量 | vLLM serve 启动参数 | 使用 `--enforce-eager`；通过 `--limit-mm-per-prompt` 将 image/video/audio 设为 0，只验证 text-only | 当前成功边界；ACLGraph和多模态尚未验收 |

### 5.3 推理与 GDN bug-fix

| ID | 原始问题 | 修改地址和符号 | 修改内容 | 当前状态 |
| --- | --- | --- | --- | --- |
| G01 | UT 因 `torch_npu` 自动加载在收集阶段失败 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/tests/ut/conftest.py` | 当 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 时强制使用 mock NPU 路径 | 只用于 UT，服务启动不能设置该变量 |
| G02 | `aclnnCausalConv1d` / GetWorkspace symbol 不存在 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/gdn.py`，`_is_missing_aclnn_causal_conv1d`、`_npu_causal_conv1d_custom_or_fallback` | eager 模式遇到明确缺 symbol 错误时回退 `_310p` PyTorch causal conv；graph capture 设置 `allow_fallback=False` | 当前工作区实际 diff；原 fatal 已越过 |
| G03 | prefill `query_start_loc` 为 `[0,0,...]` 或重复终点 | 同一 `gdn.py`，`_host_ints_to_device_tensor`、`_normalize_prefill_query_start_loc` | 单请求修为 `[0,total_tokens]`，重复终点裁剪到首个有效终点 | 当前工作区实际 diff并有 UT |
| G04 | causal conv fallback 的 split sizes 总和为 0 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/_310p/ops/causal_conv1d.py`，`causal_conv1d_fn` | 当总 token 非 0 且只有无效单序列长度时改用 `[total_tokens]` | 当前工作区实际 diff并有 UT |
| G05 | `clear_ssm_states` Triton launcher 不可用 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/triton/fla/utils.py`，`_clear_ssm_states_pytorch` | 无 Triton或已知 launcher 错误时用 mask 原地清零 | 当前工作区实际 diff并有 UT |
| G06 | `fused_gdn_gating` Triton launcher 不可用 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/device/device_op.py`，`A5DeviceAdaptor.fused_gdn_gating` | 回退 `_310p.ops.fla.fused_gdn_gating_pytorch` | 当前工作区实际 diff |
| G07 | prefill chunk GDN 的 Triton/custom ACLNN 不可用 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/gdn.py`，`_should_fallback_chunk_gated_delta_rule`、`_chunk_gated_delta_rule_or_fallback` | 只捕获 launcher 或明确缺 symbol 错误，回退 `chunk_gated_delta_rule_pytorch` | 当前工作区实际 diff |
| G08 | decode 前 `l2norm_fwd` 依赖 Triton launcher | 同一 `gdn.py`，`_l2norm_fwd_pytorch`、`_l2norm_fwd_or_fallback` | 使用 float32 `F.normalize` 后转回输入 dtype | 当前工作区实际 diff并有 UT |
| G09 | recurrent GDN 不支持 float32 state或缺 symbol | 同一 `gdn.py`，`_should_fallback_recurrent_gated_delta_rule`、`_npu_recurrent_gated_delta_rule_or_fallback` | 回退 `fused_recurrent_gated_delta_rule_pytorch`，并按原布局和 dtype 写回 state | 当前工作区实际 diff；decode 数值仍待端到端验证 |
| G10 | 上述 fallback 缺少回归保护 | `/data/xuy/qwen/vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py` | 增加错误分类、query loc、zero split、SSM、L2 norm 等测试 | 已记录定向结果 `5 passed, 14 warnings` |

### 5.4 vAscend 真实计算 bug-fix

| ID | 原始问题 | 修改地址和符号 | 修改内容 | 当前状态 |
| --- | --- | --- | --- | --- |
| R01 | no-stream 模式在 saved op 执行前返回 origin | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/autogen.py`，`REGISTER_NNOP_EXEC` | 先保存 origin 返回值，再执行 `EXECUTE_SAVED_OP`，最后返回 | Unique 动态 shape 能写回的关键修复 |
| R02 | `extra.txt` 新 ABI 与 toolkit 旧 ABI 随机去重 | 同一 `autogen.py`，`generate_file` | 由 `list(set(...))` 改为按函数名 dict 覆盖；extra 后扫描并稳定覆盖旧声明 | 可重复生成修复 |
| R03 | 所有 `aclTensor` 参数都被错误识别为 output | 同一 `autogen.py`，`FunctionGenerator`、`GRPCFunctionGenerator`、`VAscendFunctionGenerator` | 通用规则只识别非 const tensor；动态输出生成 `SetDynamicOutput`；V5 显式指定 `{46,47}` | 已显著改善，仍有 `Output count mismatch` 待逐算子清理 |
| R04 | optional `char *` 为 null 时 string 序列化不安全 | 同一 `autogen.py`，`ArgumentType.to_arguments` | `AddString` 增加 null 保护 | 有效通用修复 |
| R05 | 修改 `extra.txt` 不触发重新生成 | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/CMakeLists.txt` | custom command 增加 `${EXTRA_FILE}` 依赖 | 全量 `build.sh` 可重复构建 |
| R06 | Unique reply 数据正确但客户端输出仍是标量 | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/autogen.py`、生成客户端、`/root/vascend_nostream/simulator/config/custom_op.json` | `SetDynamicOutput`、执行顺序、`real_calculation` 和 `bad_origin` 协同修复 | 独立测试 shape `[3]`、值 `[0,1,2]` 通过 |
| R07 | ReduceSum runner 签名或 output index 不匹配 | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/NpuWorkerOperators.cpp`，`aclnnReduceSum` 注册 | 参数类型修为 tensor、IntArray、bool、dtype、output，输出索引 `{4}` | 完整模型可持续越过；独立数值测试仍待补 |
| R08 | Scatter GetWorkspace 少 4 个参数导致 AArch64 ABI 错位 | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/extra.txt` 和 `autogen.py` | 补 `cacheModeOptional`、`scatterModeOptional`、`stridesOptional`、`offsetsOptional`，完整生成 12 个算子参数 | GDB 已证明并修复 SIGSEGV 根因 |
| R09 | Scatter 的 `PA_NZ` 与 vAscend base-format tensor 不一致 | `autogen.py` 中 Scatter `AddString` 特例 | RPC 路径将 `PA_NZ` 映射为 `Norm` | 仅适用于当前 base-format 链路，不能泛化到真实 NZ tensor |
| R10 | runner 缺少 Scatter ABI和输出注册 | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/NpuWorkerOperators.cpp` | 注册 12 个类型，output indexes 为 `{1,4}` | runner 已收到并执行 Scatter |
| R11 | V5 头文件不在主 include，客户端未生成 wrapper | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/extra.txt`；真实声明源 `/usr/local/Ascend/ascend-toolkit/latest/opp/include/aclnnop/aclnn_fused_infer_attention_score_v5.h` | 补 48 个算子参数声明和输出 46/47 | 客户端已导出 V5 符号 |
| R12 | runner 未注册 V5，客户端回落本地 CANN并报 tiling key | `/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/NpuWorkerOperators.cpp` | 注册 V5 48 参数和 output indexes `{46,47}` | 原 tiling-key fatal 已消失 |
| R13 | 修改了错误的配置文件 | `/root/vascend_nostream/simulator/config/custom_op.json` | 将 Unique、Scatter、V5 加入实际生效的 `real_calculation` / `bad_origin` | 有效部署修复 |
| R14 | 多份 `libcust_opapi.so` 版本不一致 | `/root/simulator/custom_op/op_api/lib/libcust_opapi.so`、`/usr/local/lib/libcust_opapi.so`、`/data/xuy/qwen/libcust_opapi.so` | 部署后校验 SHA-256 和关键导出符号 | 有效部署修复；每次重编译后必须重复检查 |
| R15 | 只重编了客户端 SO，runner 仍是旧二进制 | 源码 `/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src`；构建根 `/data/xuy/workspaces/vAscend/cmake-build-debug` | 使用完整工程配置和已有 Protobuf/gRPC 依赖重建 runner，避免在空 `cpp_server` 子目录直接 `gmake` | runner 已出现新 Scatter 请求；部署后二进制绝对路径仍应使用 `readlink -f ./runner` 留档 |

覆盖审计结果：共整理 37 项，其中启动阶段 12 项、推理/GDN 10 项、vAscend 真实计算
15 项。历史文档中出现但最终无效的尝试没有删除，而是移入下一节并明确标记，防止后续
重复走已经证伪的路径。

### 5.5 临时诊断和未落盘项

以下内容曾帮助定位问题，但不能被同事误认为已经完成的正式修复：

- `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/worker/worker.py` 中曾尝试在 worker
  初始化阶段限制或设置编译选项；启动日志没有出现预期的 limiting 信息，该方案未成为
  最终修复。
- `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/_310p/ops/causal_conv1d.py` 中曾加入
  `[VASCEND-CONV1D-OPTION]` 和逐 kernel slice 乘加实验。它越过了 `ge.aicoreNum=64`
  的本地 `F.conv1d` 编译错误，但独立数值测试曾不一致，尚未完整回灌当前 Windows 工作区。
- `/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/worker/model_runner_v1.py` 中的
  positions CPU/NPU roundtrip 日志，以及 `device_op.py` 中的 `[VASCEND-SCATTER]`，属于
  诊断日志，性能测试前应降级或删除。
- 早期 Python `torch.library.Library(...).define(...)` 只能补 schema，不能补 NPU kernel，
  已明确不作为长期方案。
- 不全局开启 `VLLM_ENABLE_ATB_AND_DIRECT_KERNELS`；当前 A5 构建会跳过部分 direct kernel，
  强开该宏可能引入额外编译和链接问题。只把必需的 causal-conv op 注册移出条件编译。
- `VLLM_ASCEND_VECTORCORE_FALLBACK=1` 只能绕过设备属性读取，不能把普通 Python
  function 变成可启动的 Triton kernel。
- `/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/csrc/moe/causal_conv1d_v310/causal_conv1d_310_torch_adpt.h`
  曾用于核对 `aclnnCausalConv1dV310` 路径；现有证据只表明它被检查过，不表明该文件已修改
  或 V310 路径已成为最终方案。

### 5.6 启动阶段代码对照

本节与 5.2 的 `Sxx` 编号一一对应。标为“当前源码原样”的代码来自当前 Windows
工作区 diff；标为“历史修复核心”的代码来自早期运行目录和排查记录，表示当时实际采用
的关键改法，不保证行号仍与当前分支一致。代码中的 `...` 表示只省略了无关上下文。

#### S01：兼容 image preprocess 参数名

地址：

```text
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/
vllm_ascend/patch/worker/patch_qwen3vl_image_preprocess.py
```

旧调用方传入 `resample`，patch 后的处理器期望 `interpolation`。历史修复核心如下：

```python
_ORIGINAL_PATCHED_QWEN3VL_PREPROCESS = _patched_preprocess


def _patched_preprocess_compat(self, *args, **kwargs):
    if "interpolation" not in kwargs and "resample" in kwargs:
        kwargs["interpolation"] = kwargs.pop("resample")
    return _ORIGINAL_PATCHED_QWEN3VL_PREPROCESS(self, *args, **kwargs)


Qwen2VLImageProcessorFast._preprocess = _patched_preprocess_compat
```

#### S02-S03：移除 gated LayerNorm 对不可启动 Triton kernel 的硬依赖

地址：

```text
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/
vllm_ascend/ops/triton/layernorm_gated.py
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/
vllm_ascend/ops/layernorm.py
```

`S02` 先把 helper 改到稳定位置：

```python
# 旧代码依赖 vllm.triton_utils.triton.next_power_of_2。
from vllm.utils.math_utils import next_power_of_2
```

但 helper 修复后，`kernel[grid]` 仍然不可启动。因此 `S03` 的最终可执行路径是 PyTorch
fallback；历史修复核心如下：

```python
def _rms_norm_gated_torch_fallback(
    x, z, weight, eps, group_size=None, norm_before_gate=False
):
    orig_dtype = x.dtype
    x = x.float()
    weight = weight.float()
    gate = None

    if z is not None:
        gate = F.silu(z.float())
        if not norm_before_gate:
            x = x * gate

    if group_size is None:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps)
    else:
        orig_shape = x.shape
        x_group = x.reshape(*orig_shape[:-1], -1, group_size)
        variance = x_group.pow(2).mean(dim=-1, keepdim=True)
        out = (x_group * torch.rsqrt(variance + eps)).reshape(orig_shape)

    out = out * weight
    if gate is not None and norm_before_gate:
        out = out * gate
    return out.to(orig_dtype)


def forward_oot(self, x, z=None):
    return _rms_norm_gated_torch_fallback(
        x,
        z,
        self.weight,
        self.eps,
        self.group_size,
        self.norm_before_gate,
    )
```

#### S04-S05：MoE 的空 expert map 和缺失 import

地址：

```text
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/
vllm_ascend/ops/fused_moe/token_dispatcher.py
```

历史修复核心：

```python
import torch_npu

# ...
expert_map = token_dispatch_input.routing.expert_map
if expert_map is not None:
    expert_map = expert_map.npu()
```

这里不能写成无条件 `expert_map.npu()`，因为单卡或不启用 expert remap 时该值合法地为
`None`。

#### S06-S08：Qwen3.5 patch 的注册和调用边界

地址：

```text
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/
vllm_ascend/patch/worker/patch_qwen3_5.py
```

`S06` 通过 import 触发自定义 op 注册：

```python
import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope
```

`S07` 曾尝试初始化 vectorcore 属性和使用环境变量绕过属性读取，但不能解决
`kernel[grid]` 本身不可启动。最终 `S08` 不再覆盖上游 attention forward：

```python
# 保留 GDN 等仍需使用的 patch。
# 不再强制把 Qwen3NextAttention 切到不可启动的 fused Triton 路径。
# Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
```

这不是永久删除 fused attention，而是限定本阶段成功边界。恢复该赋值前必须先验证对应
Triton/custom kernel 在目标环境可启动且数值正确。

#### S09：把 causal-conv schema 和 NPU kernel 注册移出条件编译

地址：

```text
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/csrc/torch_binding.cpp
```

历史修复核心：

```cpp
ops.def(
    "npu_causal_conv1d_custom(Tensor output, Tensor x, "
    "Tensor weight, Tensor conv_state, Tensor? bias_opt, "
    "int[] query_start_loc_opt, int[] cache_indices_opt, "
    "int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, "
    "int activation_mode, int pad_slot_id, int run_mode) "
    "-> (Tensor output)");
ops.impl(
    "npu_causal_conv1d_custom",
    torch::kPrivateUse1,
    &vllm_ascend::npu_causal_conv1d_custom);
```

以上注册放在 `VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` 条件外；本阶段没有全局打开该宏。

#### S10：兼容不同 vLLM 版本的 logger import

当前源码原样，地址：

```text
D:\workspaces\vLLM-ascend_for_lingqu\vllm_ascend\logger.py
```

```python
try:
    from vllm.logging_utils import ColoredFormatter, NewLineFormatter
except ImportError:
    try:
        from vllm.logger import ColoredFormatter, NewLineFormatter
    except ImportError:

        class NewLineFormatter(logging.Formatter):

            def format(self, record):
                if not hasattr(record, "fileinfo"):
                    record.fileinfo = record.filename
                formatted = super().format(record)
                if record.message:
                    header = formatted.split(record.message, maxsplit=1)[0]
                    formatted = formatted.replace("\n", "\n" + header)
                return formatted

        class ColoredFormatter(NewLineFormatter):
            pass
```

#### S11-S12：构建环境和 text-only eager 启动边界

构建或静态检查时取消 mock preload，避免它污染工具链：

```bash
env -u LD_PRELOAD <build-or-check-command>
```

服务进程仍加载 vAscend 所需的 mock 库；两者不能混为一条通用环境配置。当前首 token
验证还要求 eager 和 text-only：

```bash
vllm serve /data/xuy/models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 2048 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
```

实际启动命令如还包含端口、内存比例或其他部署参数，应原样附在交接记录中；上面只列
与 `S12` 直接相关的限制项。

### 5.7 推理与 GDN 代码对照

本节与 5.3 的 `Gxx` 编号对应。除特别说明外，代码均为当前 Windows 工作区 diff 的
原样摘录。

#### G01：UT 显式关闭 backend autoload 时走 mock NPU

地址：

```text
D:\workspaces\vLLM-ascend_for_lingqu\tests\ut\conftest.py
```

```python
if os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0":
    _npu_available = False
else:
    try:
        subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
        _npu_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        _npu_available = False
```

该变量只用于 CPU/mock UT。真实 vLLM serve 需要 NPU backend 自动加载，不能照搬。

#### G02：causal conv 只对已知缺符号错误回退

地址：

```text
D:\workspaces\vLLM-ascend_for_lingqu\vllm_ascend\ops\gdn.py
```

错误分类和调用框架：

```python
def _is_missing_aclnn_causal_conv1d(exc: RuntimeError) -> bool:
    msg = str(exc)
    return "aclnnCausalConv1d" in msg and (
        "libopapi" in msg or "not found" in msg or "not in" in msg
    )


def _npu_causal_conv1d_custom_or_fallback(..., allow_fallback=True, ...):
    try:
        return torch.ops._C_ascend.npu_causal_conv1d_custom(...)
    except RuntimeError as exc:
        if not allow_fallback or not _is_missing_aclnn_causal_conv1d(exc):
            raise

    if run_mode == 0:
        fallback_out = causal_conv1d_fn_pytorch(...)
    else:
        fallback_out = causal_conv1d_update_pytorch(...)

    output.copy_(fallback_out)
    return output
```

省略号只隐藏了参数透传。关键约束是 graph capture 调用明确传入：

```python
_npu_causal_conv1d_custom_or_fallback(
    ...,
    allow_fallback=False,
)
```

因此 Python fallback 只用于 eager 路径，不会被误录进 graph。

#### G03-G04：规范 query 起点并修复 zero split

`G03` 位于 `vllm_ascend/ops/gdn.py`。核心规则是把单请求全零起点修为
`[0, total_tokens]`，并裁剪重复终点：

```python
def _normalize_prefill_query_start_loc(
    values, cache_indices, total_tokens, device
):
    query_start_loc = _host_ints_to_device_tensor(values, device)
    if total_tokens <= 0:
        return query_start_loc
    if query_start_loc is None:
        if cache_indices is not None and cache_indices.numel() > 0:
            return torch.tensor(
                [0, total_tokens], dtype=torch.int64, device=device
            )
        return None

    query_start_loc_cpu = query_start_loc.detach().cpu()
    if (
        query_start_loc_cpu.numel() >= 2
        and int(query_start_loc_cpu[0].item()) == 0
        and int(query_start_loc_cpu[-1].item()) == total_tokens
    ):
        total_positions = torch.nonzero(
            query_start_loc_cpu == total_tokens, as_tuple=False
        ).flatten()
        first_total_pos = int(total_positions[0].item())
        if first_total_pos > 0:
            return query_start_loc[: first_total_pos + 1]

    if query_start_loc_cpu.numel() <= 2:
        return torch.tensor([0, total_tokens], dtype=torch.int64, device=device)
    if (
        int(query_start_loc_cpu[0].item()) == 0
        and not torch.any(query_start_loc_cpu[1:] > 0).item()
    ):
        return torch.tensor([0, total_tokens], dtype=torch.int64, device=device)
    return query_start_loc
```

`G04` 位于 `vllm_ascend/_310p/ops/causal_conv1d.py`：

```python
seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
total_tokens = x.shape[-1]
if sum(seqlens) != total_tokens and total_tokens > 0:
    if len(seqlens) <= 1 or all(seq_len <= 0 for seq_len in seqlens):
        seqlens = [total_tokens]
splits = torch.split(x, seqlens, dim=-1)
```

#### G05-G06：SSM clear 和 gating 的 PyTorch fallback

`G05` 位于 `vllm_ascend/ops/triton/fla/utils.py`：

```python
def _clear_ssm_states_pytorch(ssm_states, has_initial_state):
    keep_mask = has_initial_state.to(dtype=ssm_states.dtype)
    keep_mask = keep_mask.view(
        (keep_mask.numel(),) + (1,) * (ssm_states.ndim - 1)
    )
    ssm_states.mul_(keep_mask)


if not HAS_TRITON:
    _clear_ssm_states_pytorch(ssm_states, has_initial_state)
    return

try:
    _clear_ssm_states_kernel[grid](...)
except (RuntimeError, TypeError) as exc:
    msg = str(exc)
    if (
        "function' object is not subscriptable" not in msg
        and "Device properties not initialized" not in msg
    ):
        raise
    _clear_ssm_states_pytorch(ssm_states, has_initial_state)
```

`G06` 位于 `vllm_ascend/device/device_op.py`：

```python
@staticmethod
def fused_gdn_gating(A_log, a, b, dt_bias):
    if not HAS_TRITON:
        return fused_gdn_gating_pytorch(A_log, a, b, dt_bias)
    try:
        return fused_gdn_gating_patch(A_log, a, b, dt_bias)
    except (RuntimeError, TypeError) as exc:
        if not _is_triton_launch_unavailable(exc):
            raise
        return fused_gdn_gating_pytorch(A_log, a, b, dt_bias)
```

#### G07-G09：chunk、L2 norm 和 recurrent GDN fallback

三项都位于 `vllm_ascend/ops/gdn.py`。先集中定义“允许回退”的错误，不吞掉未知错误：

```python
def _should_fallback_chunk_gated_delta_rule(exc):
    msg = str(exc)
    if _is_triton_launch_unavailable(exc):
        return True
    return (
        (
            "aclnnChunkGatedDeltaRuleFwdH" in msg
            or "aclnnChunkFwdO" in msg
        )
        and ("libopapi" in msg or "not found" in msg or "not in" in msg)
    )


def _should_fallback_recurrent_gated_delta_rule(exc):
    msg = str(exc)
    if (
        "aclnnRecurrentGatedDeltaRule" in msg
        and "params.state not implemented for DT_FLOAT" in msg
        and "DT_BFLOAT16" in msg
    ):
        return True
    return (
        "aclnnRecurrentGatedDeltaRule" in msg
        and ("libopapi" in msg or "not found" in msg or "not in" in msg)
    )
```

`G08` 的 L2 norm fallback 保留 float32 中间精度：

```python
def _l2norm_fwd_pytorch(x):
    return F.normalize(
        x.to(torch.float32), p=2, dim=-1, eps=1e-6
    ).to(x.dtype)


def _l2norm_fwd_or_fallback(x):
    if not HAS_TRITON:
        return _l2norm_fwd_pytorch(x)
    try:
        return l2norm_fwd(x)
    except (RuntimeError, TypeError) as exc:
        if not _is_triton_launch_unavailable(exc):
            raise
        return _l2norm_fwd_pytorch(x)
```

`G09` recurrent fallback 必须同时处理 state 的布局和 dtype 回写：

```python
state_for_fallback = state.transpose(-1, -2).contiguous()
fallback_out, fallback_state = fused_recurrent_gated_delta_rule_pytorch(
    q=query,
    k=key,
    v=value,
    g=g,
    beta=beta,
    initial_state=state_for_fallback,
    inplace_final_state=True,
    cu_seqlens=cu_seqlens,
    ssm_state_indices=(
        fallback_state_indices
        if fallback_state_indices is not None
        else ssm_state_indices
    ),
    num_accepted_tokens=num_accepted_tokens,
    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
)
state.copy_(
    fallback_state.transpose(-1, -2).contiguous().to(state.dtype)
)
return fallback_out
```

#### G10：回归测试入口

地址：

```text
D:\workspaces\vLLM-ascend_for_lingqu\tests\ut\ops\test_gdn_attn_builder.py
```

运行方式：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv \
  tests/ut/ops/test_gdn_attn_builder.py
```

测试必须至少覆盖以下断言语义：

```python
assert _is_missing_aclnn_causal_conv1d(missing_symbol_error)
assert not _is_missing_aclnn_causal_conv1d(unrelated_error)
assert normalized_query_start_loc.tolist() == [0, total_tokens]
torch.testing.assert_close(pytorch_l2norm, expected_l2norm)
torch.testing.assert_close(updated_ssm_state, expected_ssm_state)
```

### 5.8 vAscend 生成器和算子代码对照

本节与 5.4 的 `Rxx` 编号对应。vAscend 不在当前 Windows 仓库中，以下代码来自用户提供
的 Linux 源码、生成结果和 runner 注册输出；路径以真实 Linux 环境为准。

#### R01：no-stream 模式先执行 saved op 再返回

地址：

```text
/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/autogen.py
```

修复后的宏核心：

```cpp
#define REGISTER_NNOP_EXEC(opName)                                      \
    __attribute__((visibility("default"))) aclnnStatus opName(         \
        void *workspace, uint64_t workspaceSize,                       \
        aclOpExecutor *executor, const aclrtStream stream)              \
    {                                                                   \
        const bool isRealCalc =                                         \
            simulator::ConfigManagerRealCalc::GetInstance()             \
                .IsForRealCalc(#opName);                                \
        const bool isBadOrigin =                                        \
            simulator::ConfigManagerRealCalc::GetInstance()             \
                .IsBadOriginFunc(#opName);                              \
        const bool streamEnabled =                                      \
            simulator::ConfigManager::GetInstance().GetBoolConfig(      \
                simulator::OP_CPU_STREAM_ENABLE);                       \
        if (isRealCalc && streamEnabled) {                              \
            CLAIM_OP_TO_THREAD(executor);                               \
        }                                                               \
        aclnnStatus ret = OK;                                           \
        if (!isBadOrigin) {                                             \
            CALL_ORIGIN_OP(opName, ret);                                \
        }                                                               \
        if (isRealCalc && !streamEnabled) {                             \
            EXECUTE_SAVED_OP(executor);                                 \
        }                                                               \
        return ret;                                                     \
    }
```

关键不是宏的排版，而是 `CALL_ORIGIN_OP` 不再导致提前返回，`EXECUTE_SAVED_OP` 必须发生
在最终 `return ret` 之前。

#### R02：按函数名稳定覆盖旧 ABI

旧逻辑：

```python
all_functions.append(factory.create_generation(declaration))
# ...
all_functions = list(set(all_functions))
```

修复核心：

```python
generation = factory.create_generation(declaration)
all_functions[generation.name] = generation
```

扫描顺序保持主 include 在前、`extra.txt` 在后，因此同名声明最终由 `extra.txt` 覆盖。

#### R03-R04：输出识别和 optional string 序列化

旧输出识别过于宽泛：

```python
if "acltensor" in arg.lower():
    self.outputs.append(index)
```

修复后的生成语义是只注册非 const tensor，并为 V5 固定输出索引：

```python
if "aclTensor" in argument.arg_type and not argument.is_const:
    self.outputs.append(index)

if self.name == "aclnnFusedInferAttentionScoreV5":
    self.outputs = [46, 47]
```

上面是生成器最终语义的精简表示；实际实现可能分别位于 `FunctionGenerator`、
`GRPCFunctionGenerator` 和 `VAscendFunctionGenerator`。生成客户端时：

```python
if "aclTensorList" in arg.arg_type:
    output_lines.append(f"runner->SetTensorListOutput({arg.name});")
elif "aclTensor" in arg.arg_type and not arg.is_const:
    output_lines.append(f"runner->SetDynamicOutput({arg.name});")
```

`char *` 则统一做空指针保护：

```python
if re.match(r"(const\s+)?char\s*\*", self.arg_type):
    return (
        f'runner->AddString({self.name} == nullptr '
        f'? "" : {self.name});'
    )
```

#### R05：让 extra 声明变更触发重新生成

地址：

```text
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/CMakeLists.txt
```

```cmake
add_custom_command(
    OUTPUT ${GENERATED_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
    COMMAND ${Python3_EXECUTABLE} ${GEN_SCRIPT}
            --m vascend ${GENERATED_FILES} -e ${EXTRA_FILE}
    DEPENDS ${GEN_SCRIPT} ${EXTRA_FILE}
    COMMENT "Running Python code generator"
    VERBATIM
)
```

#### R06-R07：Unique 和 ReduceSum 输出注册

Unique runner 的确切注册：

```cpp
{"aclnnUniqueConsecutive",
 ExecuteInfo{
     .aclnnOperator =
         &AclUtils::ExecuteOnNpu<
             aclTensor *, bool, bool, int64_t, aclTensor *,
             aclTensor *, aclTensor *>,
     .outputIndexes = {4, 5, 6}}},
```

生成客户端对三个非 const 输出使用动态输出：

```cpp
runner->SetDynamicOutput(valueOut);
runner->SetDynamicOutput(inverseOut);
runner->SetDynamicOutput(countsOut);
```

ReduceSum runner 注册：

```cpp
{"aclnnReduceSum",
 ExecuteInfo{
     .aclnnOperator =
         &AclUtils::ExecuteOnNpu<
             aclTensor *, aclIntArray *, bool, aclDataType, aclTensor *>,
     .outputIndexes = {4}}},
```

#### R08-R10：Scatter 的完整 ABI、format 转换和 runner 注册

三项修改的职责边界如下：

- `R08` 修复客户端 GetWorkspace 的 12 参数 ABI。
- `R09` 只处理当前 vAscend base-format 链路中的 `PA_NZ -> Norm` 语义转换。
- `R10` 修复真实 runner 的参数类型表和 `{1, 4}` 输出索引。

`extra.txt` 中必须是 CANN 9.1 的完整 GetWorkspace 声明：

```cpp
__attribute__((visibility("default"))) aclnnStatus
aclnnScatterPaKvCacheGetWorkspaceSize(
    const aclTensor *key,
    aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional,
    char *cacheModeOptional,
    char *scatterModeOptional,
    const aclIntArray *stridesOptional,
    const aclIntArray *offsetsOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

生成客户端的四个新增参数以及 base-format 特例：

```cpp
runner->AddString(
    cacheModeOptional != nullptr &&
            std::string(cacheModeOptional) == "PA_NZ"
        ? "Norm"
        : (cacheModeOptional == nullptr ? "" : cacheModeOptional));
runner->AddString(
    scatterModeOptional == nullptr ? "" : scatterModeOptional);
runner->AddIntArray(stridesOptional);
runner->AddIntArray(offsetsOptional);
```

runner 注册必须与这 12 个算子参数完全同序：

```cpp
{"aclnnScatterPaKvCache",
 ExecuteInfo{
     .aclnnOperator =
         &AclUtils::ExecuteOnNpu<
             aclTensor *, aclTensor *, aclTensor *, aclTensor *,
             aclTensor *, aclTensor *, aclTensor *, aclTensor *,
             char *, char *, aclIntArray *, aclIntArray *>,
     .outputIndexes = {1, 4}}},
```

#### R11-R12：V5 的声明来源、客户端输出和 runner 注册

V5 声明的权威来源：

```text
/usr/local/Ascend/ascend-toolkit/latest/opp/include/aclnnop/
aclnn_fused_infer_attention_score_v5.h
```

完整声明有 48 个算子参数，随后才是 `workspaceSize` 和 `executor`。为避免在交接文档里
复制一份容易漂移的 50 参数声明，`extra.txt` 应直接以该头文件内容为准；这里保留最关键
的尾部 ABI 和输出位置：

```cpp
// ... qStartIdxOptional, kvStartIdxOptional and scalar attributes ...
const aclTensor *attentionOut,  // operator argument index 46
const aclTensor *softmaxLse,    // operator argument index 47
uint64_t *workspaceSize,
aclOpExecutor **executor);
```

生成客户端因为原 ABI 将两个输出声明为 const，需要显式注册实际输出 tensor：

```cpp
runner->SetOutput(const_cast<aclTensor *>(attentionOut));
runner->SetOutput(const_cast<aclTensor *>(softmaxLse));
```

runner 表项必须保留完整 48 参数模板列表；其不可省略的输出定义是：

```cpp
{"aclnnFusedInferAttentionScoreV5",
 ExecuteInfo{
     .aclnnOperator =
         &AclUtils::ExecuteOnNpu<
             /* 与 V5 头文件逐项同序的 48 个 ACL 参数类型 */>,
     .outputIndexes = {46, 47}}},
```

注释不能直接复制进实际模板参数列表；实际 `NpuWorkerOperators.cpp` 中必须展开全部 48 个
类型。此处只为说明索引关系，完整类型表以该源码文件为准。

### 5.9 配置、构建和部署代码对照

本节覆盖 `R13-R15`，同时给出同事可以直接执行的核验命令。

#### R13：修改真实生效的配置文件

地址：

```text
/root/vascend_nostream/simulator/config/custom_op.json
```

配置值在实际文件中是逗号分隔字符串。下面只截取本阶段新增的成员，不表示要删除原有
算子：

```json
{
  "real_calculation": "...,aclnnUniqueConsecutive,aclnnScatterPaKvCache,aclnnFusedInferAttentionScoreV5",
  "bad_origin": "...,aclnnUniqueConsecutive,aclnnScatterPaKvCache,aclnnFusedInferAttentionScoreV5"
}
```

修改后不要只目测，用 Python 精确检查成员：

```bash
python3 - <<'PY'
import json

path = "/root/vascend_nostream/simulator/config/custom_op.json"
with open(path, encoding="utf-8") as f:
    config = json.load(f)

for field in ("real_calculation", "bad_origin"):
    members = set(filter(None, config[field].split(",")))
    for op in (
        "aclnnUniqueConsecutive",
        "aclnnScatterPaKvCache",
        "aclnnFusedInferAttentionScoreV5",
    ):
        print(field, op, op in members)
PY
```

#### R14：同步并确认真正加载的客户端 SO

```bash
sha256sum \
  /root/simulator/custom_op/op_api/lib/libcust_opapi.so \
  /usr/local/lib/libcust_opapi.so \
  /data/xuy/qwen/libcust_opapi.so

nm -D /root/simulator/custom_op/op_api/lib/libcust_opapi.so | \
  grep -E 'aclnn(ScatterPaKvCache|FusedInferAttentionScoreV5)'

# 对正在运行的 EngineCore 再检查一次实际映射。
grep libcust_opapi /proc/<ENGINE_CORE_PID>/maps
```

只有构建产物存在并不代表运行进程加载了它；`/proc/<pid>/maps` 是最终判据。

#### R15：从工程根重建客户端和 runner

推荐继续使用可重复的全量构建：

```bash
cd /data/xuy/workspaces/vAscend
bash build.sh --build=Debug -g=False --build_op_mock
```

不要在一个没有完成 CMake 配置的 `cmake-build-debug/cpp_server` 空子目录直接执行
`gmake`。构建完成后记录实际启动文件：

```bash
cd <runner-deploy-directory>
readlink -f ./runner
sha256sum ./runner
stat ./runner
strings ./runner | grep -x 'aclnnScatterPaKvCache'
strings ./runner | grep -x 'aclnnFusedInferAttentionScoreV5'
```

生成文件只用于检查，不能作为持久修改点：

```text
/data/xuy/workspaces/vAscend/cmake-build-debug/src/op_cpu_mock/
cust_op/src/generated-vascend/libcust_opapi.cpp
```

持久修改必须落在 `autogen.py`、`extra.txt`、CMake 源文件或 runner 源码中。

## 6. vLLM Ascend 侧修改

### 6.1 GDN 运行时 fallback

主要文件：

```text
vllm_ascend/ops/gdn.py
vllm_ascend/_310p/ops/causal_conv1d.py
vllm_ascend/ops/triton/fla/utils.py
vllm_ascend/device/device_op.py
```

主要修改：

- 识别 `aclnnCausalConv1d` 缺失并回退到 PyTorch 实现。
- 修复 prefill 中异常的 `query_start_loc=[0,0,...]`。
- 修复 zero-length sequence 导致的 `torch.split` 失败。
- 给 chunk gated delta rule 增加无 Triton fallback。
- 给 recurrent gated delta rule 增加缺符号和 state dtype fallback。
- 给 L2 norm 增加无 Triton fallback。
- 给 `clear_ssm_states` 和 `fused_gdn_gating` 增加 PyTorch fallback。
- graph capture 路径保留严格模式，不允许静默进入 Python fallback。

这些改动的目的不是性能优化，而是在当前 Triton/custom ACLNN 不完整的环境中保持
prefill/decode 语义可执行，并让真实计算排查能够继续向后推进。

### 6.2 causal conv1d 调试结论

最初的 `_310p` fallback 使用：

```python
F.conv1d(...)
```

在当前模拟环境中会触发：

```text
Value 64 for parameter ge.aicoreNum is invalid.
value must be in range [0, 4].
```

曾尝试在 `F.conv1d` 前设置编译选项，但错误未消失。随后在运行容器中尝试使用
逐 kernel slice 的乘加实现，成功越过该 CANN 编译错误。

注意：

- 该运行容器 patch 尚未完整回灌到当前 Windows 工作区。
- 独立数值测试曾出现输出全为 0 或 1，与 CPU 参考值不一致。
- 当前只能确认模型已越过原始 Conv2D/`ge.aicoreNum` 错误，不能据此声明数值正确。

### 6.3 日志和调试观测

曾加入以下观测信息：

```text
DEBUG CPU positions expected=...
DEBUG positions CPU-NPU roundtrip=...
[VASCEND-SCATTER] key=... key_cache=... slots=...
[VASCEND-CONV1D-OPTION] ...
```

这些日志用于确认 tensor shape、CPU/NPU roundtrip 和算子入口。它们属于诊断性改动，
后续性能验证前应降级或删除。

### 6.4 单元测试

当前工作区新增或修改：

```text
tests/ut/conftest.py
tests/ut/ops/test_gdn_attn_builder.py
```

覆盖内容包括：

- causal conv1d 缺符号错误分类；
- Triton launcher 不可用错误分类；
- chunk/recurrent GDN fallback；
- recurrent state dtype 不支持；
- prefill `query_start_loc` 修复；
- zero-length causal conv1d 修复；
- clear SSM state、L2 norm 和 gating fallback。

已记录的定向测试结果：

```text
5 passed, 14 warnings
```

## 7. vAscend 客户端生成器修改

权威源码位于：

```text
/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/autogen.py
/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/extra.txt
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/CMakeLists.txt
```

生成文件位于：

```text
/data/xuy/workspaces/vAscend/cmake-build-debug/src/op_cpu_mock/cust_op/src/generated-vascend/libcust_opapi.cpp
```

### 7.1 修复执行顺序

`REGISTER_NNOP_EXEC` 原先可能在执行 saved op 前直接返回 origin，导致 gRPC 返回的动态
shape/data 没有及时写回调用方。

修复后的 no-stream 逻辑为：

1. 判断 `isRealCalc`、`isBadOrigin` 和 stream mode。
2. stream mode 下先 claim executor。
3. origin 可用时调用 origin，并保存返回值。
4. no-stream real calculation 下执行 saved op。
5. 最后返回 origin 或 mock 状态。

这项修改是 `aclnnUniqueConsecutive` shape 能正确回写的关键前提。

### 7.2 extra 声明覆盖旧头文件

生成器原先使用：

```python
all_functions.append(...)
all_functions = list(set(all_functions))
```

这只能按函数名去重，但无法保证工具包旧声明和 `extra.txt` 新声明谁胜出。

现改为按函数名保存：

```python
all_functions[generation.name] = generation
```

扫描顺序为：

1. `$ASCEND_HOME_PATH/include/aclnnop`
2. `extra.txt`

因此 `extra.txt` 中的新 ABI 会稳定覆盖主 include 目录里的旧 ABI。

### 7.3 输出索引识别

原生成器把所有包含 `aclTensor` 的参数都当作 output，包括 const 输入 tensor。

现调整为：

- 普通算子只把非 const `aclTensor` 参数识别为 output。
- `aclnnFusedInferAttentionScoreV5` 特例指定 `{46,47}`。
- V5 客户端使用 `SetOutput(const_cast<aclTensor *>(...))` 注册
  `attentionOut` 和 `softmaxLse`。

这同时改善了 runner output index 的准确性。

### 7.4 char 指针和 Scatter cache mode

所有 `char *` 参数增加空指针保护：

```cpp
runner->AddString(value == nullptr ? "" : value);
```

Scatter 在当前 vAscend 客户端只保留 ND/base format。若直接把 `cacheMode="PA_NZ"`
发送给真实 runner，CANN 会按 FRACTAL_NZ 校验 cache 物理布局并失败。

因此对 Scatter 特殊处理：

```cpp
cacheModeOptional == "PA_NZ" ? "Norm" : cacheModeOptional
```

该转换只适用于当前 vAscend base-format 数据链路，不应无条件推广到真实 PA_NZ tensor。

### 7.5 CMake 生成依赖

原 CMake 自定义命令只依赖：

```cmake
DEPENDS ${GEN_SCRIPT}
```

现改为：

```cmake
DEPENDS ${GEN_SCRIPT} ${EXTRA_FILE}
```

这样修改 `extra.txt` 后能够自动重新生成 `libcust_opapi.cpp`。

### 7.6 构建方式

现在可以继续使用全量构建：

```bash
cd /data/xuy/workspaces/vAscend
bash build.sh --build=Debug -g=False --build_op_mock
```

因为修复已进入 `autogen.py`、`extra.txt` 和 CMake 源码，全量构建即使清理
`cmake-build-debug`，也会重新生成正确代码。

禁止继续手工修改：

```text
cmake-build-debug/.../generated-vascend/libcust_opapi.cpp
```

手工修改会在下一次生成时丢失。

## 8. 关键 ACLNN 算子修复

### 8.1 aclnnUniqueConsecutive

原始现象：

```text
runner reply valueOut shape=[3], data=[0,1,2]
client actual shape=[]
```

根因：

- 动态输出没有在正确时间写回。
- executor 执行顺序导致 saved op 未在函数返回前完成。
- origin/mock 路径组合不适合该动态 shape 算子。

修改：

- 为输出 tensor 使用动态输出注册。
- 修复 `REGISTER_NNOP_EXEC` 的执行顺序。
- 将 `aclnnUniqueConsecutive` 加入 `real_calculation` 和 `bad_origin`。
- 同步并验证真实生效的 `libcust_opapi.so`。

独立测试结果：

```text
shape: (3,)
result: [0, 1, 2]
PASS
```

### 8.2 aclnnReduceSum

早期 runner 侧 ReduceSum 注册/参数签名不匹配，导致完整模型中反复出现该算子时无法
稳定执行。已校正 runner 调用签名和输出映射，后续日志中大量
`Got request: aclnnReduceSum` 能持续执行。

该算子目前属于“链路已越过”，但仍建议补充独立数值回归用例。

### 8.3 aclnnScatterPaKvCache

#### ABI 根因

旧生成声明只有 8 个算子参数：

```text
key, keyCacheRef, slotMapping, value, valueCacheRef,
compressLensOptional, compressSeqOffsetOptional, seqLensOptional
```

CANN 9.1 实际还包含：

```text
cacheModeOptional
scatterModeOptional
stridesOptional
offsetsOptional
```

由于缺少这 4 个参数，AArch64 调用约定把：

```text
workspaceSize -> "PA_NZ"
executor      -> "None"
```

GDB 证据：

```text
x/s workspaceSize = "PA_NZ"
x/s executor      = "None"
SIGSEGV in UniqueExecutor::ReleaseTo
```

#### 修复

- 在 `extra.txt` 中加入准确的 12 参数 GetWorkspace 声明。
- 客户端 RPC 增加两个 string 和两个 IntArray。
- runner 注册 12 个参数类型。
- runner output indexes 设置为 `{1,4}`。
- 加入 `real_calculation` 和 `bad_origin`。
- 将 vAscend base format 下的 `PA_NZ` 转换为 `Norm`。

runner 注册的核心形式：

```cpp
{"aclnnScatterPaKvCache",
 ExecuteInfo{
     .aclnnOperator =
         &AclUtils::ExecuteOnNpu<
             aclTensor *, aclTensor *, aclTensor *, aclTensor *,
             aclTensor *, aclTensor *, aclTensor *, aclTensor *,
             char *, char *, aclIntArray *, aclIntArray *>,
     .outputIndexes = {1, 4}}},
```

验证进展：

1. 最初独立调用为 `std::bad_alloc`。
2. 修正 bad origin 后暴露 ABI 错位 SIGSEGV。
3. ABI 修复后 runner 收到 12 参数请求。
4. 修正 PA_NZ/ND 语义后，完整 vLLM 日志出现：

```text
Got request: aclnnScatterPaKvCache
```

### 8.4 aclnnFusedInferAttentionScoreV5

#### 原始问题

客户端本地执行原生 CANN V5，报错：

```text
OP [FusedInferAttentionScore] not find tilingKey[266608898].
```

当时 runner 没有收到 V5 请求，说明该算子未被 vAscend 截获。

#### 头文件差异

主 include 目录只稳定提供 V2/V3/V4，而 V5 头文件位于真实 runner 的：

```text
/usr/local/Ascend/ascend-toolkit/latest/opp/include/aclnnop/
aclnn_fused_infer_attention_score_v5.h
```

V5 GetWorkspace 共 50 个参数：

- 48 个算子参数；
- `workspaceSize`；
- `executor`。

输出索引：

```text
46 = attentionOut
47 = softmaxLse
```

#### 修复

- 将准确 V5 声明加入 `extra.txt`。
- 生成客户端 GetWorkspace wrapper 和执行函数。
- V5 强制输出索引 `{46,47}`。
- 客户端注册两个 const output tensor。
- runner 注册 48 个参数类型。
- 将 V5 加入 `real_calculation` 和 `bad_origin`。
- 同步所有可能被动态加载的 `libcust_opapi.so`。

当前状态：

- 活动 SO 已导出 V5 GetWorkspace 和执行符号。
- 修正实际配置文件后，本地 tiling-key fatal error 不再出现。
- 最新 `max_tokens=1` 请求已完成真实 NPU 首 token、HTTP 200 和流式收尾，期间没有出现
  新的 V5 错误。

## 9. 真实 NPU runner 修改

runner 关键入口：

```text
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/
NpuWorkerOperators.cpp
```

执行模板：

```cpp
template<typename ...AclArgs>
void ExecuteOnNpu(
    const std::string &aclOpName,
    const aclrtStream &stream,
    const std::vector<AclPrimitive_t> &args)
```

本阶段 runner 侧主要工作：

- 核对 proto string、list、IntArray 的解码方式。
- 补齐 Scatter 12 参数注册和 `{1,4}` 输出。
- 补齐 FusedInferAttentionScoreV5 48 参数注册和 `{46,47}` 输出。
- 根据真实 CANN 9.1 头文件校正函数签名。
- 使用 request/reply 日志确认算子确实进入真实 NPU 执行。

注意 runner 二进制和客户端 SO 必须来自同一套声明。任一侧仍使用旧 ABI，都可能表现为
`_Map_base::at`、`std::bad_alloc`、SIGSEGV 或无意义的 CANN 参数错误。

## 10. 配置和部署

### 10.1 实际生效的配置文件

客户端启动使用：

```bash
SIMULATOR_HOME=/root/vascend_nostream
```

因此真实生效的配置是：

```text
/root/vascend_nostream/simulator/config/custom_op.json
```

不是其他同名目录下的 `custom_op.json`。

关键字段：

```json
{
  "real_calculation": "...aclnnScatterPaKvCache,...aclnnUniqueConsecutive,...aclnnFusedInferAttentionScoreV5",
  "bad_origin": "...aclnnScatterPaKvCache,...aclnnUniqueConsecutive,...aclnnFusedInferAttentionScoreV5"
}
```

`bad_origin` 用于避免客户端本地 CANN 对当前环境不支持或 ABI 不一致的算子先行执行。

### 10.2 libcust_opapi.so 一致性

至少存在三份可能影响运行的 SO：

```text
/root/simulator/custom_op/op_api/lib/libcust_opapi.so
/usr/local/lib/libcust_opapi.so
/data/xuy/qwen/libcust_opapi.so
```

部署后必须检查：

```bash
sha256sum \
  /root/simulator/custom_op/op_api/lib/libcust_opapi.so \
  /usr/local/lib/libcust_opapi.so \
  /data/xuy/qwen/libcust_opapi.so
```

并确认关键符号：

```bash
nm -D --defined-only /root/simulator/custom_op/op_api/lib/libcust_opapi.so |
grep -E 'aclnn(ScatterPaKvCache|FusedInferAttentionScoreV5)'
```

此前曾出现：

- custom-op 目录是新 SO；
- `/usr/local/lib` 仍是旧 SO；
- V5 配置修改到了错误的 `custom_op.json`。

这会让部分算子进入 vAscend，另一些算子仍绑定到原生 `libopapi.so`，表现非常迷惑。

## 11. 当前验证证据

| 验证项 | 结果 | 结论边界 |
| --- | --- | --- |
| vLLM 启动 | 通过 | API Server 和 EngineCore 可用 |
| positions CPU/NPU roundtrip | 通过 | 当前样例 positions 一致 |
| Unique 独立测试 | 通过 | 动态 shape 和数据正确 |
| GDN 原始阻塞点 | 已越过 | fallback 数值仍需系统验证 |
| Scatter 独立/完整链路 | 已进入 runner | cache 写入数值仍需检查 |
| V5 本地 tiling error | 已消失 | 需确认 runner reply 和最终 attention 数值 |
| 完整模型无新 fatal | 通过 | 一次请求已完成 |
| HTTP 流式响应 | 通过 | HTTP 200、`finish_reason="length"`、`[DONE]` |
| 首 token 生成链路 | 通过 | prefill、真实计算、sampling、流式收尾全部完成 |
| token 内容识别 | 待验证 | 尚未开启 `return_token_ids`，可见 `content` 为空 |
| 输出质量 | 未验收 | 不能仅凭空字符串判断 logits 或语义正确 |
| 性能 | 不通过 | 当前链路过慢 |

### 11.1 真实 NPU 首 token 端到端证据

请求采用最小 prompt 和单 token 上限：

```json
{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": "Hi"}],
  "temperature": 0,
  "max_tokens": 1,
  "stream": true
}
```

关键响应：

```text
data: {"id":"chatcmpl-808de71cf552514b",...,"delta":{"role":"assistant","content":""},...}
data: {"id":"chatcmpl-808de71cf552514b",...,"delta":{"content":""},"finish_reason":"length",...}
data: [DONE]
```

API Server 和调度器同时记录：

```text
127.0.0.1:59630 - "POST /v1/chat/completions HTTP/1.1" 200 OK
Running: 0 reqs, Waiting: 0 reqs
```

本次 prompt 共 11 个 token，positions CPU/NPU roundtrip 为 `0..10`。日志中反复出现
`[VASCEND-SCATTER] key=(11,2,256)`，随后进入 sampler 并正常结束。`max_tokens=1`
时首 token 直接从 prefill 末尾 logits 采样，不要求再执行一次 token 长度为 1 的 decode
forward，因此没有看到 `key=(1,2,256)` 不能被判定为 decode 故障。

由此可以确认 API Server、EngineCore、模型 prefill、真实 NPU runner、sampler 和流式响应
收尾均已连通，真实 NPU 首 token 生成链路已经跑通。由于请求未启用
`return_token_ids`，且默认可能过滤特殊 token，空 `content` 仍不能证明 logits、采样结果
或模型语义正确，也不能反向否定首 token 执行链路已经完成。

## 12. 当前性能瓶颈

这次请求最终返回，进一步排除了“tensor 计算达到固定上限后永久卡住”的判断。窗口持续
出现新请求时，计算确实仍在推进。

主要慢点包括：

- Qwen3.5-35B-A3B 完整 40 层 prefill。
- no-stream 模式下的大量串行执行。
- `ASCEND_LAUNCH_BLOCKING=1` 强制同步。
- 大量细粒度 ACLNN 算子逐个通过 gRPC 往返。
- tensor 编码、复制和输出回写。
- runner 的逐算子 REQUEST/REPLY/DebugString 打印。
- vLLM 侧 fallback 和 `Output count mismatch` 日志。
- Python/PyTorch fallback 无法使用目标设备上的高性能 fused kernel。

调试阶段建议使用：

```bash
curl -N --max-time 1800 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0,
    "max_tokens": 1,
    "stream": true
  }'
```

即使只生成一个 token，首轮 prefill 仍必须执行全部模型层。

### 12.1 首 token 端到端耗时观测

本次响应的 `created` 时间戳对应：

```text
2026-07-29 14:24:43 +08:00
```

日志在约 `14:37:25` 进入 sampler，并在随后完成响应，粗略可见首个生成步耗时约
13 分钟。由于现有日志没有记录 curl 发起时刻、首字节和 `[DONE]` 的精确时间，该数字
只作为阶段性观测，后续应使用 `curl -w` 或客户端计时分别记录 TTFT 和总耗时。

完成时 vLLM 打印：

```text
Avg prompt throughput: 1.1 tokens/s
Avg generation throughput: 0.1 tokens/s
Running: 0 reqs
Waiting: 0 reqs
```

`Running: 0` 和 `Waiting: 0` 进一步说明请求正常离开调度器，不是超时后遗留的挂起请求。

## 13. 已知遗留问题

### 13.1 Output count mismatch

当前仍能看到：

```text
Output count mismatch after parsing: parsed=0 registered=1
Output count mismatch: result tensor index=0, registered outputs=0
```

生成器已修复“所有 tensor 都被视为 output”的通用问题，但仍可能存在：

- runner 使用了旧生成结果；
- 个别 in-place/ref 算子的 output index 需要特例；
- server reply 对无数据 output 的处理不一致；
- optional output 没有根据实际开关调整。

该警告当前未造成 fatal，但必须在数值验收前清理。

交接排查入口：

```bash
grep -Rsn --binary-files=without-match \
  -E 'Output count mismatch|registered outputs|after parsing' \
  /data/xuy/workspaces/vAscend/src \
  /data/xuy/workspaces/vAscend/cmake-build-debug \
  2>/dev/null
```

日志至少应补充 op name、客户端 registered indexes、runner output indexes、reply tensor
count 和 result tensor index。当前只有数量，无法判断哪些告警属于无输出/in-place 算子，
哪些已经造成真实 tensor 未回写。

### 13.2 数值一致性

以下路径需要独立 CPU/NPU 对照测试：

- causal conv1d；
- Scatter cache 写入；
- FusedInferAttentionScoreV5；
- recurrent/chunk GDN fallback；
- in-place 算子的 output reply。

### 13.3 代码归档

vAscend 修改目前主要位于 Linux 编译环境，必须形成正式 diff/commit。运行容器中的临时
Conv1D patch 和诊断日志也需要判断是否回灌。

## 14. 下一阶段计划

建议按以下顺序继续：

1. 使用 `return_token_ids=true`、`skip_special_tokens=false` 和 `logprobs=true` 复测
   `max_tokens=1`，确认空文本对应的实际 token。
2. 将 `max_tokens` 提升到 2，确认出现 token 长度为 1 的 decode forward，并验证 KV cache
   写回能支撑第二个 token。
3. 给所有 `Output count mismatch` 日志增加算子名、registered indexes 和 reply tensor count，
   逐算子清理客户端/runner 输出映射。
4. 关闭或降级 runner 的逐 tensor DebugString，保留算子名、耗时、请求/响应字节和错误。
5. 为 runner 增加算子计数和累计耗时，定位 TTFT 的主要耗时来源。
6. 为 Unique、Scatter、V5、Conv1D 增加独立数值回归。
7. 对比真实计算结果与 CPU/PyTorch reference。
8. 减少 gRPC 往返、完整 cache 回传和 host/device copy。
9. 逐步恢复可用的 fused/custom kernel，减少 Python fallback。
10. 完成至少一个短请求的非空、合理输出质量验证。
11. 将 vLLM Ascend 和 vAscend 两侧修改分别提交并记录 commit hash。

## 15. 复现检查清单

### 15.1 构建 vAscend

```bash
cd /data/xuy/workspaces/vAscend
bash build.sh --build=Debug -g=False --build_op_mock
```

### 15.2 检查生成结果

```bash
GEN=/data/xuy/workspaces/vAscend/cmake-build-debug/src/op_cpu_mock/cust_op/src/generated-vascend/libcust_opapi.cpp

grep -n 'REGISTER_NNOP_EXEC(aclnnFusedInferAttentionScoreV5)' "$GEN"
grep -n 'cacheModeOptional.*scatterModeOptional.*stridesOptional.*offsetsOptional' "$GEN"
```

### 15.3 检查配置

```bash
CFG=/root/vascend_nostream/simulator/config/custom_op.json

python3 - "$CFG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
targets = {
    "aclnnUniqueConsecutive",
    "aclnnScatterPaKvCache",
    "aclnnFusedInferAttentionScoreV5",
}

for field in ("real_calculation", "bad_origin"):
    values = {v.strip() for v in config[field].split(",") if v.strip()}
    print(field, {name: name in values for name in targets})
PY
```

### 15.4 检查 SO

```bash
sha256sum \
  /root/simulator/custom_op/op_api/lib/libcust_opapi.so \
  /usr/local/lib/libcust_opapi.so \
  /data/xuy/qwen/libcust_opapi.so
```

### 15.5 启动 runner

```bash
readlink -f ./runner
./runner
```

预期：

```text
Server listening on 0.0.0.0:50043
```

交接记录必须保存 `readlink -f` 的结果和 `sha256sum ./runner`。当前聊天只记录了
`x00957222` 用户目录下执行 `./runner`，未留下该二进制的绝对部署地址和 hash，不能在
文档中臆造。

### 15.6 最小请求

```bash
curl -N --max-time 1800 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":1,"stream":true}'
```

首 token 内容诊断请求：

```bash
curl -sS --max-time 1800 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.5",
    "messages":[{"role":"user","content":"Hi"}],
    "temperature":0,
    "max_tokens":1,
    "stream":false,
    "skip_special_tokens":false,
    "return_token_ids":true,
    "logprobs":true,
    "top_logprobs":5
  }'
```

## 16. 文件清单

### 16.1 当前 vLLM Ascend 工作区

```text
/data/xuy/qwen/vLLM-ascend_for_lingqu/tests/ut/conftest.py
/data/xuy/qwen/vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py
/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/_310p/ops/causal_conv1d.py
/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/device/device_op.py
/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/logger.py
/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/gdn.py
/data/xuy/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/triton/fla/utils.py
```

早期启动阶段还修改过：

```text
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/csrc/torch_binding.cpp
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/fused_moe/token_dispatcher.py
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/layernorm.py
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/triton/layernorm_gated.py
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/patch/worker/patch_qwen3_5.py
/data/xuy/workspace/qwen/vLLM-ascend_for_lingqu/vllm_ascend/patch/worker/patch_qwen3vl_image_preprocess.py
```

这些早期路径下的改动需要与当前 `/data/xuy/qwen/vLLM-ascend_for_lingqu` 逐项比对，不能
仅凭历史文档假定已经全部回灌。

### 16.2 vAscend 源码

明确修改并需要提交的文件：

```text
/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/autogen.py
/data/xuy/workspaces/vAscend/src/op_cpu_mock/common/extra.txt
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/CMakeLists.txt
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/NpuWorkerOperators.cpp
```

关联实现和诊断检查点，不应在没有 diff 证据时表述为正式修改：

```text
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/libcust_opapi.h
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/op_cpu_sim/inc/CpuOpRunner.h
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cust_op/src/op_cpu_sim/src/OpReqExecutor.cpp
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/main.cpp
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/NpuWorker.h
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/NpuWorker.cpp
/data/xuy/workspaces/vAscend/src/op_cpu_mock/cpp_server/src/AclPrimitives/AclTensor.cpp
```

其中 `main.cpp` 曾用于打印指定算子的完整 REQUEST/REPLY；`CpuOpRunner` 和
`OpReqExecutor` 负责 AddString/AddIntArray、output 注册和 reply 回写；`NpuWorker` 中的
`ExecuteOnNpu` 负责把 proto primitive 恢复为 CANN 参数。它们是继续定位 mismatch 的
首要入口。

### 16.3 运行时配置和产物

```text
/root/vascend_nostream/simulator/config/custom_op.json
/root/simulator/custom_op/op_api/lib/libcust_opapi.so
/usr/local/lib/libcust_opapi.so
/data/xuy/qwen/libcust_opapi.so
```

## 17. 阶段性评价

过去一个月的工作已经把问题从“模型无法稳定进入真实计算”推进到“完整 Qwen3.5 请求
通过 vAscend 真实 NPU runner 完成一次 prefill 和首 token 生成，并正常结束 HTTP 流式
响应”。
期间解决了动态输出、执行顺序、AArch64 ABI、tensor format、runner 注册、生成器覆盖、
配置路径和动态库版本不一致等多个彼此叠加的问题。

当前最重要的成果不是性能，而是：

- 真实计算链路已经建立；
- 已知 fatal error 已逐个被消除；
- 第一次端到端真实计算首 token 请求已经返回 HTTP 200、`finish_reason="length"` 和
  `[DONE]`；
- 关键修复已经从生成文件上移到可重复构建的源码；
- 后续问题已经从“首 token 能否执行完”转向“token 是什么、输出是否正确、decode 是否
  连续、如何提速”。

下一阶段应以“识别实际 token ID并完成第二个 token 的 decode”为第一里程碑，然后取得
非空、合理输出，同时开始数值一致性验证和性能优化。
