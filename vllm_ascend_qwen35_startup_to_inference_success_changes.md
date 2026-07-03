# vLLM-Ascend Qwen3.5 从启动成功到推理跑通的改动总览

> 日期：2026-07-03  
> 模型：`/data/xuy/models/Qwen3.5-35B-A3B/`  
> 服务名：`qwen3.5`  
> vLLM 版本：`0.22.1`  
> 运行形态：Ascend simulator / hook 环境，当前未部署真实 NPU 计算环境与 GRPC 计算后端。

本文整理从“vLLM 服务可以启动，但首个推理请求失败”到“`/v1/chat/completions` 接口完整返回”的全部关键改动点。

注意：当前输出内容仍不正常，例如短请求返回 `"!!!!"`，说明生成质量和数值一致性还需要继续验证。但从服务闭环角度看，HTTP API、调度、prefill、decode、采样和 OpenAI-compatible 返回链路已经跑通。

---

## 1. 最终状态

### 1.1 服务状态

当前服务已完成：

```text
GET /health       200 OK
GET /v1/models    200 OK
POST /v1/chat/completions  200 OK
```

最终 smoke 请求：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"你好，回答两个字"}],"temperature":0,"max_tokens":4}'
```

返回摘要：

```json
{
  "object": "chat.completion",
  "model": "qwen3.5",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "!!!!"
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 4,
    "total_tokens": 18
  }
}
```

这说明：

- API server 可访问。
- 模型名匹配。
- tokenizer / scheduler / prefill / decode / sampler / response serialization 均已走通。
- 请求不再因为 GDN、Triton fallback、ACLNN symbol 或 dtype 支持问题崩溃。

### 1.2 当前仍需注意

- 输出质量尚不正常，后续需要继续做数值正确性和真实计算后端验证。
- 当前 simulator / fallback 路径吞吐很低，日志约为 `1 token/s`。
- `max_tokens=128` 的非流式请求会超过 `test_vllm_service.py` 默认 `timeout=120`，这不是接口未通，而是生成太慢。
- 未部署真实 NPU 计算环境与 GRPC 时，多个算子走 PyTorch fallback 或仿真路径，性能和数值都不能代表最终真实环境。

---

## 2. 运行环境和启动参数

关键环境变量：

```bash
LD_PRELOAD=/usr/local/lib/libvnnopbase.so:/usr/local/lib/libhccl_mock.so:/usr/local/lib/libdlopen_vllm.so
VASCEND=1
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
ASCEND_CUSTOM_OPP_PATH=/root/simulator/custom_op:
ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.1.0
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
```

启动命令形态：

```bash
vllm serve /data/xuy/models/Qwen3.5-35B-A3B/ \
  --trust-remote-code \
  --host 0.0.0.0 \
  --served-model-name qwen3.5 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enforce-eager
```

当前关键限制：

- 使用 `--enforce-eager`，所以 cudagraph / compile 路径没有作为最终成功路径验证。
- 多模态输入被限制为 0，当前验证的是 text-only 推理链路。
- Triton 在当前环境不可用或不可 launch，日志中会出现：

```text
Triton not installed or not compatible; certain GPU-related functions will not be available.
TypeError: 'function' object is not subscriptable
```

---

## 3. 改动总览

### 3.1 启动阶段已有改动

以下改动主要来自已有记录：

- `vllm_ascend_qwen35_debug_patches.md`
- `vllm_ascend_qwen35_success_startup_changes.md`

这些改动使服务从“启动阶段失败”推进到“API Server 正常启动，`/health` 和 `/v1/models` 返回 200”。

| 序号 | 问题 | 修改方向 | 涉及文件 |
| --- | --- | --- | --- |
| 1 | Qwen3VL image preprocess 参数不兼容，`interpolation` / `resample` 签名不一致 | 给 `_patched_preprocess` 增加兼容 wrapper | `vllm_ascend/patch/worker/patch_qwen3vl_image_preprocess.py` |
| 2 | `triton.next_power_of_2` 缺失 | 从 `vllm.utils.math_utils` 补 `next_power_of_2` 或改为直接调用 helper | `vllm_ascend/ops/triton/layernorm_gated.py` 等 |
| 3 | `layernorm_gated.py` Triton kernel 不可 launch | 在 `LayerNorm` 路径中增加 PyTorch fallback | `vllm_ascend/ops/layernorm.py` |
| 4 | 单卡 / 单 rank 场景 `expert_map=None` | MoE token dispatcher 兼容 `expert_map=None` | `vllm_ascend/ops/fused_moe/token_dispatcher.py` |
| 5 | MoE 路径中 `torch_npu` 未导入 | 补充必要 import | MoE 相关文件 |
| 6 | `triton_split_qkv_rmsnorm_mrope` op 未注册 | 补 import，确保 custom op 注册路径被执行 | 相关 `__init__` / op 注册文件 |
| 7 | `split_qkv_rmsnorm_mrope.py` vectorcore 初始化异常 | 给 `core_num` 等信息增加兜底 | `vllm_ascend/ops/triton/split_qkv_rmsnorm_mrope.py` |
| 8 | Qwen3.5 fused attention monkey patch 在当前环境不可用 | 禁用不适合当前环境的 fused attention forward，保留必要 patch | `vllm_ascend/patch/worker/patch_qwen3_5.py` |
| 9 | Python 侧 schema patch 不适合作为长期方案 | 改为 C++ 注册 `npu_causal_conv1d_custom` | C++ custom op 注册相关文件 |
| 10 | vLLM 日志模块跨版本 import 差异 | 对 `ColoredFormatter` / `NewLineFormatter` 增加 import fallback | `vllm_ascend/logger.py` |

启动阶段最终状态：

```text
Application startup complete.
GET /health HTTP/1.1 200 OK
GET /v1/models HTTP/1.1 200 OK
```

但当时首个推理请求仍失败，主要卡在 GDN / CANN / Triton 相关运行时路径。

### 3.2 推理阶段新增改动

以下是本轮把服务从“启动正常，推理失败”推进到“启动和推理均跑通”的主要新增改动。

| 文件 | 改动目的 |
| --- | --- |
| `tests/ut/conftest.py` | 允许 UT 在 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 下跳过 `torch_npu` 后端自动加载，避免测试收集阶段失败 |
| `vllm_ascend/ops/gdn.py` | 给 GDN prefill / decode 关键路径增加多层 fallback，覆盖 causal conv1d、chunk GDN、L2 norm、recurrent GDN |
| `vllm_ascend/_310p/ops/causal_conv1d.py` | 修复 query_start_loc 异常导致的 zero-length split 问题 |
| `vllm_ascend/ops/triton/fla/utils.py` | 给 `clear_ssm_states` 增加 PyTorch fallback |
| `vllm_ascend/device/device_op.py` | 给 `fused_gdn_gating` 增加 PyTorch fallback |
| `tests/ut/ops/test_gdn_attn_builder.py` | 增加针对上述 fallback 和真实错误签名的回归测试 |

---

## 4. 推理阶段问题和修复链路

### 4.1 单元测试收集失败：`torch_npu` 自动加载

问题：

```text
RuntimeError: Failed to load the backend extension: torch_npu.
You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
```

原因：

当前容器中 `npu-smi` 存在，`tests/ut/conftest.py` 会判断 NPU 可用，进而触发真实 `torch_npu` 路径。但 UT 只需要 mock NPU，不应该因为后端自动加载失败而无法收集测试。

修改：

```python
if os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0":
    _npu_available = False
else:
    ...
```

影响：

- 只影响 UT 环境。
- 服务启动不要设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

验证：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
  vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py::test_gdn_runtime_fallback_error_classification \
  -q
```

### 4.2 `aclnnCausalConv1d` 缺失或不可用

问题：

```text
RuntimeError: aclnnCausalConv1d or aclnnCausalConv1dGetWorkspaceSize not in libopapi.so
```

原因：

当前 CANN / simulator 环境中的 `libopapi.so` 不包含所需 ACLNN symbol，或者 custom OPP 不完整。原 GDN 路径直接调用：

```python
torch.ops._C_ascend.npu_causal_conv1d_custom(...)
```

一旦底层 symbol 缺失，请求直接 fatal。

修改：

在 `vllm_ascend/ops/gdn.py` 中增加：

- `_is_missing_aclnn_causal_conv1d`
- `_npu_causal_conv1d_custom_or_fallback`
- `_causal_conv1d_activation`
- `_host_ints_to_device_tensor`
- `_normalize_prefill_query_start_loc`

当 eager 路径遇到 `aclnnCausalConv1d` 缺失时，回落到 `_310p` PyTorch 实现：

```python
causal_conv1d_fn_pytorch(...)
causal_conv1d_update_pytorch(...)
```

同时保留图捕获路径的严格行为：

```python
allow_fallback=False
```

这样当前 `--enforce-eager` 能走 fallback，未来 graph path 不会悄悄混入 Python fallback。

### 4.3 prefill 阶段 `query_start_loc` 异常导致 split 失败

问题：

```text
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 42
but got split_sizes=[0]
```

原因：

prefill fallback 收到的 `query_start_loc` 形如 `[0, 0]` 或等价的全零序列，但实际输入有 42 个 token。PyTorch fallback 根据 query_start_loc 计算 `seqlens=[0]`，最终 `torch.split` 失败。

修改：

在 `vllm_ascend/ops/gdn.py` 中新增 `_normalize_prefill_query_start_loc`：

- 如果 total tokens 大于 0，但 query_start_loc 只有 0 或全部非正，则修复为 `[0, total_tokens]`。
- 如果 query_start_loc 形如 `[0, 42, 42, 42]`，裁剪为 `[0, 42]`。

在 `vllm_ascend/_310p/ops/causal_conv1d.py` 中增加兜底：

```python
total_tokens = x.shape[-1]
if sum(seqlens) != total_tokens and total_tokens > 0:
    if len(seqlens) <= 1 or all(seq_len <= 0 for seq_len in seqlens):
        seqlens = [total_tokens]
```

效果：

- prefill causal conv1d fallback 不再因为 zero-length split 崩溃。
- 单请求 prefill 可以继续往后进入 GDN attention。

### 4.4 `clear_ssm_states` Triton kernel 不可 launch

问题：

```text
TypeError: 'function' object is not subscriptable
```

原因：

当前环境中 Triton 不可用，`_clear_ssm_states_kernel[grid](...)` 中的 kernel 是普通 Python function，不支持 Triton launch 语法。

修改文件：

```text
vllm_ascend/ops/triton/fla/utils.py
```

新增 PyTorch fallback：

```python
def _clear_ssm_states_pytorch(ssm_states, has_initial_state):
    keep_mask = has_initial_state.to(dtype=ssm_states.dtype)
    keep_mask = keep_mask.view((keep_mask.numel(),) + (1,) * (ssm_states.ndim - 1))
    ssm_states.mul_(keep_mask)
```

在 `HAS_TRITON=False` 或 Triton launch 抛出已知错误时回落。

### 4.5 `fused_gdn_gating` Triton kernel 不可 launch

问题：

```text
TypeError: 'function' object is not subscriptable
Device properties not initialized
```

原因：

GDN gating 原先调用 Triton fused kernel。当前 simulator 环境没有可用 Triton driver。

修改文件：

```text
vllm_ascend/device/device_op.py
```

新增：

- `_is_triton_launch_unavailable`
- `fused_gdn_gating_pytorch` fallback

逻辑：

```python
if not HAS_TRITON:
    return fused_gdn_gating_pytorch(...)
try:
    return fused_gdn_gating_patch(...)
except (RuntimeError, TypeError) as exc:
    if not _is_triton_launch_unavailable(exc):
        raise
    return fused_gdn_gating_pytorch(...)
```

### 4.6 prefill `chunk_gated_delta_rule` 不可用

问题：

可能出现：

```text
TypeError: 'function' object is not subscriptable
RuntimeError: aclnnChunkFwdO ... not in libopapi.so
```

原因：

prefill GDN chunk 路径优先走 Triton / Ascend custom kernel，但当前环境同时存在 Triton 不可 launch 和部分 ACLNN symbol 缺失风险。

修改：

在 `vllm_ascend/ops/gdn.py` 中新增：

- `_should_fallback_chunk_gated_delta_rule`
- `_chunk_gated_delta_rule_or_fallback`

fallback 到：

```python
chunk_gated_delta_rule_pytorch(...)
```

影响：

- prefill 阶段可以从 custom / Triton kernel 不可用场景继续推进。
- 仍保留非已知错误的抛出，避免把 shape 或数值 bug 静默吞掉。

### 4.7 decode 阶段 `l2norm_fwd` Triton kernel 不可 launch

问题：

```text
File ".../vllm/model_executor/layers/fla/ops/l2norm.py", line 117, in l2norm_fwd
    l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](...)
TypeError: 'function' object is not subscriptable
```

原因：

decode GDN 在 recurrent attention 前直接调用 upstream `l2norm_fwd`，该实现依赖 Triton launch。

修改：

在 `vllm_ascend/ops/gdn.py` 中新增：

```python
def _l2norm_fwd_pytorch(x):
    return F.normalize(x.to(torch.float32), p=2, dim=-1, eps=1e-6).to(x.dtype)

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

替换 decode 分支：

```python
query_spec = _l2norm_fwd_or_fallback(query_spec)
key_spec = _l2norm_fwd_or_fallback(key_spec)
query_non_spec = _l2norm_fwd_or_fallback(query_non_spec)
key_non_spec = _l2norm_fwd_or_fallback(key_non_spec)
```

效果：

- decode 阶段越过 `l2norm_fwd` Triton launcher 错误。

### 4.8 decode 阶段 recurrent GDN state dtype 不支持

问题：

```text
RuntimeError: call aclnnRecurrentGatedDeltaRule failed, detail:
Tensor params.state not implemented for DT_FLOAT, should be in dtype support list [DT_BFLOAT16,].
```

原因：

当前 `ssm_state` 是 float32，而底层 `aclnnRecurrentGatedDeltaRule` 只支持 bf16 state。直接调用 ACLNN 会 fatal。

修改：

在 `vllm_ascend/ops/gdn.py` 中扩展 `_should_fallback_recurrent_gated_delta_rule`：

```python
if (
    "aclnnRecurrentGatedDeltaRule" in msg
    and "params.state not implemented for DT_FLOAT" in msg
    and "DT_BFLOAT16" in msg
):
    return True
```

并通过 `_npu_recurrent_gated_delta_rule_or_fallback` 回落到：

```python
fused_recurrent_gated_delta_rule_pytorch(...)
```

注意：

- 没有强行把 state cast 成 bf16，因为那会改变状态精度和更新语义。
- 只对明确的 ACLNN dtype 支持错误进行 fallback。

### 4.9 非流式请求 timeout 的判断

现象：

`test_vllm_service.py` 默认：

```text
max_tokens=128
timeout=120
stream=False
```

服务日志：

```text
Avg generation throughput: 1.0 tokens/s
Running: 1 reqs
```

解释：

- 这不是推理链路未通，而是 simulator / fallback 路径生成太慢。
- 非流式请求必须等完整 `max_tokens` 生成结束才返回。
- 128 tokens 以约 1 token/s 生成，容易超过 120 秒 timeout。

验证方式：

```bash
python3 test_vllm_service.py --max-tokens 8 --timeout 300 --no-stream
```

或使用短 curl：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"你好，回答两个字"}],"temperature":0,"max_tokens":4}'
```

---

## 5. 回归测试

新增或扩展的测试位于：

```text
tests/ut/ops/test_gdn_attn_builder.py
```

覆盖内容：

- `aclnnCausalConv1d` 缺 symbol 错误分类。
- Triton launcher 不可用错误分类。
- `aclnnChunkFwdO` / chunk GDN fallback 错误分类。
- recurrent GDN 缺 symbol 错误分类。
- recurrent GDN `params.state DT_FLOAT` 不支持错误分类。
- prefill `query_start_loc=[0,0]` 和 `[0,42,42,42]` 修复。
- causal conv1d fallback zero-length split 修复。
- `clear_ssm_states` 无 Triton fallback。
- `l2norm_fwd` 无 Triton和 launcher TypeError fallback。

用户容器已验证过的命令示例：

```bash
cd /data/xuy/qwen
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest \
  vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py::test_gdn_runtime_fallback_error_classification \
  vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py::test_prefill_query_start_loc_fallback_repairs_single_request \
  vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py::test_causal_conv1d_fallback_repairs_zero_length_single_request \
  vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py::test_clear_ssm_states_falls_back_without_triton \
  vLLM-ascend_for_lingqu/tests/ut/ops/test_gdn_attn_builder.py::test_gdn_l2norm_falls_back_when_triton_launcher_is_unavailable \
  -q
```

已见结果：

```text
5 passed, 14 warnings in 0.10s
```

本地静态检查已执行：

```bash
python -m py_compile \
  vllm_ascend/ops/gdn.py \
  tests/ut/ops/test_gdn_attn_builder.py \
  vllm_ascend/ops/triton/fla/utils.py \
  vllm_ascend/_310p/ops/causal_conv1d.py \
  vllm_ascend/device/device_op.py \
  tests/ut/conftest.py
```

结果：通过。

---

## 6. 当前工作区相关文件

本轮推理跑通链路的主要代码文件：

```text
tests/ut/conftest.py
tests/ut/ops/test_gdn_attn_builder.py
vllm_ascend/_310p/ops/causal_conv1d.py
vllm_ascend/device/device_op.py
vllm_ascend/ops/gdn.py
vllm_ascend/ops/triton/fla/utils.py
```

启动阶段已有文档中记录的主要文件：

```text
vllm_ascend/patch/worker/patch_qwen3vl_image_preprocess.py
vllm_ascend/ops/layernorm.py
vllm_ascend/ops/fused_moe/token_dispatcher.py
vllm_ascend/ops/triton/layernorm_gated.py
vllm_ascend/ops/triton/split_qkv_rmsnorm_mrope.py
vllm_ascend/patch/worker/patch_qwen3_5.py
C++ custom op 注册相关文件
vllm_ascend/logger.py
```

---

## 7. 阶段性结论

本轮改动的核心思路是：

1. 启动阶段：通过兼容 HF processor、关闭或绕过当前环境不可用的 fused / Triton / MoE 路径，使模型能加载，API server 能启动。
2. 推理阶段：围绕 Qwen3.5 GDN 线性注意力路径逐个补 fallback，使 prefill 和 decode 不再依赖当前环境缺失或不可 launch 的 ACLNN / Triton kernel。
3. 验证阶段：用短 `max_tokens` 请求确认 OpenAI-compatible chat completion 能完整返回。

当前可以认为：

- 服务启动已跑通。
- text-only 推理接口已跑通。
- 当前输出质量尚不正常，但不影响“服务链路已跑通”的判断。
- 后续重点应从“接口能不能返回”转向“真实 NPU / GRPC 计算后端接入、fallback 数值一致性、输出质量和性能恢复”。

---

## 8. 建议后续验证

### 8.1 短请求 smoke test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"请只回答：你好"}],"temperature":0,"max_tokens":8}'
```

### 8.2 避免非流式长输出误判 timeout

```bash
python3 test_vllm_service.py --max-tokens 8 --timeout 300 --no-stream
```

### 8.3 接入真实计算环境后重新验证

需要重点观察：

- `Avg generation throughput` 是否显著高于 simulator / fallback 的约 `1 token/s`。
- 输出是否从重复标点恢复为合理文本。
- GDN fallback 是否仍被频繁触发。
- 是否可以逐步恢复 custom op / Triton / graph 路径。

