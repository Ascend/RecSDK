# 【MeetUp】TorchRec-V1 多级缓存 DCNv2 总结报告

**双卡实测完成，稳态吞吐提升 10.73%。**

## 1. 环境与工作负载

| 项目 | 实机记录 |
| --- | --- |
| GitCode ID | KxPlaug |
| 日期 | 2026-11-20 |
| HiDevLab 实例 | `M1120_J08_DCNv2` |
| 镜像 | 【应用使能】RecSDK 26.1.0 |
| 设备 | Atlas 800T A2，2 × Ascend 910B3，每卡 HBM 64 GiB |
| 卡号 | `npu-smi` 物理编号 4、5，TorchNPU 容器内逻辑编号 0、1 |
| 软件 | CANN 9.1.0-beta.3，Torch 2.7.1+cpu，TorchNPU 2.7.1.post4，Python 3.11.0 |
| 算法 | DLRM-DCNv2，TorchRec-V1 EmbCache，双卡 row-wise 分表 |
| 数据 | 固定种子合成 Criteo 格式，24 天各 262,144 条。全局 batch 16,384 |
| 版本 | Meta DLRM `b631a99cf5ff320272d18a776cda85b5207bdf19`，应用 RecSDK 官方 NPU 补丁 |

实例创建时间与所用镜像见 [平台实例记录](01_env_create_record.png)。资源规格、登录、双卡健康状态和版本分别见 [资源信息](02_resource_config.png)、[WebIDE](03_env_login.png)、[npu-smi](04_npu_smi.png) 和 [依赖预检](05_dual_npu_preflight.png)。

## 2. 实际执行

```bash
./run.sh check
./run.sh data
./run.sh smoke
./run.sh profile --variant baseline --train-batches 64 --warmup-steps 32 --measure-steps 24 --prof-start 40 --prof-steps 8
./run.sh profile --variant tuned --train-batches 64 --warmup-steps 32 --measure-steps 24 --prof-start 40 --prof-steps 8
./run.sh benchmark --train-batches 64 --warmup-steps 32 --measure-steps 24 --repeats 2
```

短跑完成 79 次训练迭代，测试 AUROC 为 0.499911，计时窗口吞吐 15,825.43 samples/s。合成标签对应的 AUROC 仅用于验证评估流程。配置的步数上限为 80，官方样例内部的迭代截断使训练实际完成 79 步。 [合成数据清单](12_dataset_manifest.png) · [短跑结果](07_smoke_completed.png)

## 3. Profiling 与调优

两组各采集逻辑设备 0、1 的第 40–47 步。基线的跨卡阻塞呈明显不对称。一侧 `Communication(Not Overlapped)` 中位数为 554.25 ms，另一侧为 47.49 ms。后者的 Host 事件 `embedding_lookup_async get in initiate_swap_in` 8 次共 4,452.98 ms。基线第 40 步的一次 AllToAllV 算子经过约 726 ms，HCCS 传输约 23 ms，长跨度主要包含等待。

调优组开启 `DO_EC_LOCAL_UNIQUE=1` 和 `ENABLE_PARALLEL_GLOBAL_UNIQUE=1`。模型、数据、卡数、全局 batch 保持一致。耗时较长的一侧在两组中换了 rank，下面按两卡中的较大值比较。Host 自耗时是采样内累计值，逐步指标是各卡八步的中位数。

| 指标 | 基线 | 调优 |
| --- | ---: | ---: |
| `Stage` 中位数较大的一侧 | 648.13 ms | 572.83 ms |
| 非重叠通信中位数较大的一侧 | 554.25 ms | 527.36 ms |
| `gloo:all_to_all` 累计 Host 自耗时最大值 | 1,577.60 ms | 932.70 ms |
| Embedding 查表等待累计 Host 自耗时最大值 | 4,452.98 ms | 4,633.99 ms |
| 开 Profiling 时的窗口吞吐 | 12,426.77 samples/s | 13,400.55 samples/s |

调优组采样中，Gloo 通信累计自耗时下降，Embedding 查表等待仍然突出。开 Profiling 时的吞吐仅作为诊断信息，验收看关闭 Profiling 的配对测试。 [基线完成](09_baseline_profile_complete.png) · [基线 CSV 样本](10_baseline_profiler_csv.png) · [调优计时窗口](11_tuned_profiler_csv.png)

## 4. 吞吐验收

**通过。** 基线中位数 16,410.32 samples/s，调优中位数 18,171.61 samples/s，提升 **10.73%**，达到任务书 10% 的目标。每轮固定 32 个预热步，再测 24 个实际完成的训练步，每次计入 393,216 条样本。用双卡中较慢一侧的耗时计算全局吞吐。第 1 轮先跑基线，第 2 轮先跑调优。

| 轮次 | 基线 samples/s | 调优 samples/s | 本轮提升 | 测试 AUROC，基线 / 调优 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16,192.16 | 18,475.35 | +14.10% | 0.500465 / 0.500322 |
| 2 | 16,628.48 | 17,867.86 | +7.45% | 0.495600 / 0.501451 |

[无 Profiling 对照截图](13_benchmark_comparison.png) · [验收退出输出](14_training_completion.png) · [脱敏比较数据](comparison.json)

## 5. 发现的问题

上游样例在 mmap 读取合成输入时提示 NumPy 数组不可写，训练和评估继续完成。Profiler 采用跳过前 40 步、热身步数取 0 的上游配置，启动时给出 warmup 提示。性能验收另行关闭 Profiling 测量。

日志中的名义迭代数比实际完成的迭代多 1，已在 80→79 和 64→63 的真机结果中复核。该问题已有社区记录。本次验收计时代码只使用实际完成的训练步。 [RecSDK Issue #1460](https://gitcode.com/Ascend/RecSDK/issues/1460)

## 6. 体验总结

镜像预装的 TorchRec-V1 与 CANN 可直接启动双卡任务。停留最久的阶段是每个训练进程首次触发算子和填充缓冲区，初期迭代耗时较高。建议样例文档同时给出预热后的计时区间、实际迭代数和每卡/全局吞吐口径，并给出双卡 EmbCache 的 trace 分析示例。
