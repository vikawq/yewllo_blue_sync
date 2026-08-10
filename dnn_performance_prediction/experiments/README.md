# DNN 性能预测实验复现手册

本文件只负责环境、命令和产物目录。统一结论、论文调研和实验解释见[主报告](../README.md)；第一、第二阶段原始叙事分别保留在[阶段一归档](../phase1_archive.md)和[阶段二归档](../phase2_archive.md)。

所有实验均依赖作者发布的 profile、预测器、opgraph、标签或 microbenchmark 数据。本机没有目标 NVIDIA GPU 或 Cortex-A76，因此不要把这些命令的成功解释为论文整体精度复现。

## 目录与正式产物

```text
D:\workspaces\trace\.research\upstream\vidur
D:\workspaces\trace\.research\upstream\nn-Meter
D:\workspaces\trace\.research\upstream\NeuSight
D:\workspaces\trace\.research\envs\vidur
D:\workspaces\trace\.research\envs\nnmeter-compat
D:\workspaces\trace\.research\envs\neusight-wsl
D:\workspaces\trace\survey\dnn_performance_prediction\experiments\results
```

正式结果目录：

- `results/vidur_tp1/2026-08-10_11-41-25-825419`
- `results/vidur_tp2/2026-08-10_11-41-50-915691`
- `results/graybox_calibration`
- `D:\workspaces\trace\.research\experiments\neusight-wsl\gpt3-h100`
- `D:\workspaces\trace\.research\experiments\neusight-wsl\bmm-train`

`results/graybox_calibration_debug` 是开发期 2-seed 干跑，不能用于正式结论。

## 1. Vidur CPU-only 事件模拟

在本目录运行相同 workload 的 TP=1、TP=2：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_vidur.ps1 -TensorParallelSize 1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_vidur.ps1 -TensorParallelSize 2
```

脚本使用仓库自带的 LLaMA-2-7B/A100 compute 与 network profile，禁用 WandB 网络上报，并将请求级 CSV、配置和 Chrome trace 写入：

```text
results/vidur_tp<N>/<timestamp>/
```

本轮环境使用 Python 3.12.13；官方 README 推荐 Python 3.10，这是已记录的环境偏差。当前没有完整环境 lock，editable finder 的映射也不自包含；`run_vidur.ps1` 会先进入 upstream 仓库，因此现有重跑入口可用。已有两组结果的统一指标见 [`results/summary.csv`](results/summary.csv)。

## 2. nn-Meter 预训练预测器

运行官方 MobileNetV3-Small IR：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_nnmeter.ps1
```

这里的 `Bypass` 只作用于新启动的 PowerShell 进程，不修改系统级执行策略。预期输出：

```text
[RESULT] predict latency for mobilenetv3small_0.json: 12.558942703135 ms
```

兼容环境：

```text
Python 3.11.15
numpy 1.26.4
scipy 1.11.4
pandas 2.1.4
scikit-learn 1.2.2
setuptools 80.10.2
```

官方 `cortexA76cpu_tflite21 v1.0` 预测器首次使用会下载约 376 MB 到 workspace 外的 `C:\Users\admin\.nn_meter\data\predictor`。其 pickle 由 scikit-learn 0.23.1 生成；1.2.2 可以读取但会告警，较新版本可能因 tree dtype 不兼容失败。本轮结果已进入 `results/summary.csv`，但没有单独保存原始 CLI 日志。

重建环境：

```powershell
uv python install 3.11
uv venv --python 3.11 D:\workspaces\trace\.research\envs\nnmeter-compat
uv pip install --python D:\workspaces\trace\.research\envs\nnmeter-compat\Scripts\python.exe `
  setuptools==80.10.2 numpy==1.26.4 scipy==1.11.4 pandas==2.1.4 scikit-learn==1.2.2 `
  -e D:\workspaces\trace\.research\upstream\nn-Meter
```

## 3. NeuSight WSL artifact

完整环境创建、CPU shim、命令、锁定依赖和结果解释见：

[`D:\workspaces\trace\.research\experiments\neusight-wsl\README.md`](../../../.research/experiments/neusight-wsl/README.md)

在 WSL 发行版 `Ubuntu-22.04-VPN` 中重跑 GPT-3 2.7B/H100 artifact：

```powershell
wsl -d Ubuntu-22.04-VPN -- bash /mnt/d/workspaces/trace/.research/experiments/neusight-wsl/run_artifact.sh
```

跑通官方 BMM 一轮 CPU 训练链路：

```powershell
wsl -d Ubuntu-22.04-VPN -- bash /mnt/d/workspaces/trace/.research/experiments/neusight-wsl/run_bmm_train.sh
```

锁定文件和正式结果：

- [`requirements-lock.txt`](../../../.research/experiments/neusight-wsl/requirements-lock.txt)
- [`run_artifact.sh`](../../../.research/experiments/neusight-wsl/run_artifact.sh)
- [`run_bmm_train.sh`](../../../.research/experiments/neusight-wsl/run_bmm_train.sh)
- [`NeuSight prediction JSON`](../../../.research/experiments/neusight-wsl/gpt3-h100/out/prediction/NVIDIA_H100_80GB_HBM3/neusight/gpt3_27-inf-2048-2.json)
- [`Roofline prediction JSON`](../../../.research/experiments/neusight-wsl/gpt3-h100/out/prediction/NVIDIA_H100_80GB_HBM3/roofline/gpt3_27-inf-2048-2.json)

## 4. Docker 灰盒 OOD 与分段校准

从本目录构建镜像并运行 10-seed、严格 shape-OOD 的 FP32 Linear/GEMM 对照：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_graybox_docker.ps1
```

脚本使用：

- [`Dockerfile.graybox`](Dockerfile.graybox)
- [`requirements-graybox.txt`](requirements-graybox.txt)
- [`graybox_calibration.py`](graybox_calibration.py)

正式结果：

- [`zero_shot.csv`](results/graybox_calibration/zero_shot.csv)
- [`calibration_runs.csv`](results/graybox_calibration/calibration_runs.csv)
- [`calibration_summary.csv`](results/graybox_calibration/calibration_summary.csv)
- [`metadata.json`](results/graybox_calibration/metadata.json)
- [`zero_shot_mape.png`](results/graybox_calibration/zero_shot_mape.png)
- [`calibration_mape.png`](results/graybox_calibration/calibration_mape.png)
- [`DETERMINISM.md`](results/graybox_calibration/DETERMINISM.md)

镜像摘要、输入/代码哈希、单线程确定性设置和两次复跑的输出哈希均记录在 `DETERMINISM.md`。该核验仅证明同一已构建镜像重复运行一致；基础镜像没有按 digest 固定，未来重建不承诺 bitwise 一致。

## 5. 结果等级

| 实验 | 等级 | 不能据此声称 |
| --- | --- | --- |
| Vidur | 发布 profile 上的 pipeline smoke | A100 真机 `<9%` 已复现，或 TP=2 普遍更慢 |
| nn-Meter | 发布预测器 smoke | Cortex-A76 实际 latency 或 ±10% 精度已验证 |
| NeuSight GPT-3/H100 | 发布 opgraph/权重/标签的单例 artifact 对齐 | 已重新测量 H100 或复现论文整体指标 |
| NeuSight BMM 1 epoch | CPU 训练入口 smoke | 模型已收敛或达到论文精度 |
| Docker 灰盒 | 作者发布真实 GPU microbenchmark 上的探索性 post-kernel 对照 | pre-kernel tactic、端到端 SLA 或新 GPU 真机精度已证明 |
