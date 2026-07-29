# Qwen3.5-35B-A3B 接入 vAscend 真实计算阶段总结

> 日期：2026-07-29  
> 模型：`/data/xuy/models/Qwen3.5-35B-A3B/`  
> vLLM：`0.22.1`  
> 服务名：`qwen3.5`  
> 当前阶段：真实 NPU runner 已进入完整模型计算链路，暂无新的 fatal error，但首个 HTTP token 尚未返回。

## 1. 文档范围

本文记录近一个月内，为把 Qwen3.5-35B-A3B 从本地 simulator/fallback 路径推进到
vAscend gRPC 真实 NPU 计算路径所做的关键修改、验证证据和遗留问题。

此前文档中记录过 simulator-only 模式下接口能够返回，但输出内容和数值并不可信。
本阶段的目标不同：

1. vLLM 在客户端容器正常启动。
2. ACLNN GetWorkspace 和执行函数被 `libcust_opapi.so` 正确截获。
3. 需要真实计算的算子通过 gRPC 发送给 NPU runner。
4. runner 使用真实 CANN `libopapi.so` 执行算子并返回 tensor。
5. 模型 forward 能持续越过已经发现的阻塞算子。

本文中的“当前无新错误”不等同于“推理最终验收通过”。最终验收仍要求 HTTP 200、
非空且合理的模型输出，以及基础数值一致性验证。

## 2. 当前结论

### 2.1 已达到的状态

- vLLM API Server 和 EngineCore 可以正常启动。
- text-only 请求已进入模型首轮 prefill。
- `aclnnUniqueConsecutive` 动态输出 shape 问题已解决，独立测试通过。
- GDN/causal-conv1d 原始阻塞点已被越过。
- `aclnnScatterPaKvCache` 已从客户端成功发送到真实 NPU runner。
- `aclnnFusedInferAttentionScoreV5` 已加入客户端生成代码、runner 注册和真实计算配置。
- 最新运行没有出现新的 CANN、ABI、segmentation fault 或 `std::bad_alloc` 错误。
- vLLM 和 runner 仍持续产生算子输出，说明请求仍在计算链路中推进。

### 2.2 尚未达到的状态

- curl 尚未收到首个生成 token。
- 尚未获得一次完整的真实计算 HTTP 200 响应。
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
| 当前 | 请求持续运行但 curl 未返回 | 无新 fatal error，主要问题转为性能和完成性 |

## 5. vLLM Ascend 侧修改

### 5.1 GDN 运行时 fallback

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

### 5.2 causal conv1d 调试结论

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

### 5.3 日志和调试观测

曾加入以下观测信息：

```text
DEBUG CPU positions expected=...
DEBUG positions CPU-NPU roundtrip=...
[VASCEND-SCATTER] key=... key_cache=... slots=...
[VASCEND-CONV1D-OPTION] ...
```

这些日志用于确认 tensor shape、CPU/NPU roundtrip 和算子入口。它们属于诊断性改动，
后续性能验证前应降级或删除。

### 5.4 单元测试

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

## 6. vAscend 客户端生成器修改

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

### 6.1 修复执行顺序

`REGISTER_NNOP_EXEC` 原先可能在执行 saved op 前直接返回 origin，导致 gRPC 返回的动态
shape/data 没有及时写回调用方。

修复后的 no-stream 逻辑为：

1. 判断 `isRealCalc`、`isBadOrigin` 和 stream mode。
2. stream mode 下先 claim executor。
3. origin 可用时调用 origin，并保存返回值。
4. no-stream real calculation 下执行 saved op。
5. 最后返回 origin 或 mock 状态。

这项修改是 `aclnnUniqueConsecutive` shape 能正确回写的关键前提。

### 6.2 extra 声明覆盖旧头文件

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

### 6.3 输出索引识别

原生成器把所有包含 `aclTensor` 的参数都当作 output，包括 const 输入 tensor。

现调整为：

- 普通算子只把非 const `aclTensor` 参数识别为 output。
- `aclnnFusedInferAttentionScoreV5` 特例指定 `{46,47}`。
- V5 客户端使用 `SetOutput(const_cast<aclTensor *>(...))` 注册
  `attentionOut` 和 `softmaxLse`。

这同时改善了 runner output index 的准确性。

### 6.4 char 指针和 Scatter cache mode

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

### 6.5 CMake 生成依赖

原 CMake 自定义命令只依赖：

```cmake
DEPENDS ${GEN_SCRIPT}
```

现改为：

```cmake
DEPENDS ${GEN_SCRIPT} ${EXTRA_FILE}
```

这样修改 `extra.txt` 后能够自动重新生成 `libcust_opapi.cpp`。

### 6.6 构建方式

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

## 7. 关键 ACLNN 算子修复

### 7.1 aclnnUniqueConsecutive

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

### 7.2 aclnnReduceSum

早期 runner 侧 ReduceSum 注册/参数签名不匹配，导致完整模型中反复出现该算子时无法
稳定执行。已校正 runner 调用签名和输出映射，后续日志中大量
`Got request: aclnnReduceSum` 能持续执行。

该算子目前属于“链路已越过”，但仍建议补充独立数值回归用例。

### 7.3 aclnnScatterPaKvCache

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

### 7.4 aclnnFusedInferAttentionScoreV5

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
- 最新完整推理持续执行，没有出现新的 V5 错误。

## 8. 真实 NPU runner 修改

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

## 9. 配置和部署

### 9.1 实际生效的配置文件

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

### 9.2 libcust_opapi.so 一致性

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

## 10. 当前验证证据

| 验证项 | 结果 | 结论边界 |
| --- | --- | --- |
| vLLM 启动 | 通过 | API Server 和 EngineCore 可用 |
| positions CPU/NPU roundtrip | 通过 | 当前样例 positions 一致 |
| Unique 独立测试 | 通过 | 动态 shape 和数据正确 |
| GDN 原始阻塞点 | 已越过 | fallback 数值仍需系统验证 |
| Scatter 独立/完整链路 | 已进入 runner | cache 写入数值仍需检查 |
| V5 本地 tiling error | 已消失 | 需确认 runner reply 和最终 attention 数值 |
| 完整模型无新 fatal | 当前通过 | 请求仍在运行 |
| 首 token | 未完成 | curl 尚未返回 |
| 输出质量 | 未验证 | 不能声明真实推理成功 |
| 性能 | 不通过 | 当前链路过慢 |

## 11. 当前性能瓶颈

目前没有证据表明存在“tensor 计算上限”。窗口持续出现新请求说明计算仍在推进。

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

## 12. 已知遗留问题

### 12.1 Output count mismatch

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

### 12.2 数值一致性

以下路径需要独立 CPU/NPU 对照测试：

- causal conv1d；
- Scatter cache 写入；
- FusedInferAttentionScoreV5；
- recurrent/chunk GDN fallback；
- in-place 算子的 output reply。

### 12.3 代码归档

vAscend 修改目前主要位于 Linux 编译环境，必须形成正式 diff/commit。运行容器中的临时
Conv1D patch 和诊断日志也需要判断是否回灌。

## 13. 下一阶段计划

建议按以下顺序继续：

1. 关闭或降级 runner 的逐 tensor DebugString，保留算子名、耗时和错误。
2. 为 runner 增加算子计数和耗时统计，判断首个 prefill 卡在哪一类算子。
3. 使用 `max_tokens=1` 完成首次真实计算 HTTP 返回。
4. 清理所有 `Output count mismatch`。
5. 为 Unique、Scatter、V5、Conv1D 增加独立数值回归。
6. 对比真实计算结果与 CPU/PyTorch reference。
7. 减少 gRPC 往返和 host/device copy。
8. 逐步恢复可用的 fused/custom kernel，减少 Python fallback。
9. 完成至少一个短请求的输出质量验证。
10. 将 vLLM Ascend 和 vAscend 两侧修改分别提交并记录 commit hash。

## 14. 复现检查清单

### 14.1 构建 vAscend

```bash
cd /data/xuy/workspaces/vAscend
bash build.sh --build=Debug -g=False --build_op_mock
```

### 14.2 检查生成结果

```bash
GEN=/data/xuy/workspaces/vAscend/cmake-build-debug/src/op_cpu_mock/cust_op/src/generated-vascend/libcust_opapi.cpp

grep -n 'REGISTER_NNOP_EXEC(aclnnFusedInferAttentionScoreV5)' "$GEN"
grep -n 'cacheModeOptional.*scatterModeOptional.*stridesOptional.*offsetsOptional' "$GEN"
```

### 14.3 检查配置

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

### 14.4 检查 SO

```bash
sha256sum \
  /root/simulator/custom_op/op_api/lib/libcust_opapi.so \
  /usr/local/lib/libcust_opapi.so \
  /data/xuy/qwen/libcust_opapi.so
```

### 14.5 启动 runner

```bash
./runner
```

预期：

```text
Server listening on 0.0.0.0:50043
```

### 14.6 最小请求

```bash
curl -N --max-time 1800 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":1,"stream":true}'
```

## 15. 文件清单

### 15.1 当前 vLLM Ascend 工作区

```text
tests/ut/conftest.py
tests/ut/ops/test_gdn_attn_builder.py
vllm_ascend/_310p/ops/causal_conv1d.py
vllm_ascend/device/device_op.py
vllm_ascend/logger.py
vllm_ascend/ops/gdn.py
vllm_ascend/ops/triton/fla/utils.py
```

### 15.2 vAscend 源码

```text
src/op_cpu_mock/common/autogen.py
src/op_cpu_mock/common/extra.txt
src/op_cpu_mock/cust_op/src/CMakeLists.txt
src/op_cpu_mock/cust_op/src/libcust_opapi.h
src/op_cpu_mock/cpp_server/src/NpuWorkerOperators.cpp
```

### 15.3 运行时配置和产物

```text
/root/vascend_nostream/simulator/config/custom_op.json
/root/simulator/custom_op/op_api/lib/libcust_opapi.so
/usr/local/lib/libcust_opapi.so
/data/xuy/qwen/libcust_opapi.so
```

## 16. 阶段性评价

过去一个月的工作已经把问题从“模型无法稳定进入真实计算”推进到“完整 Qwen3.5 请求
正在通过 vAscend 真实 NPU runner 持续执行”。期间解决了动态输出、执行顺序、AArch64
ABI、tensor format、runner 注册、生成器覆盖、配置路径和动态库版本不一致等多个彼此
叠加的问题。

当前最重要的成果不是性能，而是：

- 真实计算链路已经建立；
- 已知 fatal error 已逐个被消除；
- 关键修复已经从生成文件上移到可重复构建的源码；
- 后续问题已经从“能否执行”转向“何时完成、数值是否正确、如何提速”。

下一阶段应以“拿到首个真实计算 token”为第一里程碑，再进入数值一致性和性能优化。
