# 性能优化总览

本文档汇集昇腾性能调优全流程以及相应的工具、方法，供开发者进行性能数据采集、分析、调优参考。

## 快速导航：场景决策表

根据您的场景和瓶颈类型，快速定位到对应章节：

| 场景 | 瓶颈类型 | 现象特征 | 跳转章节 |
|------|---------|---------|---------|
| **训练** | 不确定瓶颈在哪 | 整体性能不达预期，需系统排查 | [总体策略](#总体策略) → [性能调优流程](#性能调优流程) |
| **训练** | Host CPU瓶颈 | CPU利用率高、算子下发延迟大 | [Host优化 - 通用方法](#通用方法) → [PyTorch host调优](#pytorch-host调优) |
| **训练** | Host内存瓶颈 | 内存分配耗时高、页面置换频繁 | [Host优化 - 通用方法](#通用方法) |
| **训练** | Host编译瓶颈 | 模型编译耗时长、首轮迭代慢 | [PyTorch host调优](#pytorch-host调优) |
| **训练** | Device计算瓶颈 | NPU利用率低、算子耗时集中 | [PyTorch device调优](#pytorch-device调优) / [TensorFlow device调优](#tensorflow-device调优) |
| **训练** | Device通信瓶颈 | 通信耗时占比高、多卡扩展效率低 | [通信优化](#通信优化) |
| **训练** | Device IO瓶颈 | 数据加载耗时高、NPU等待数据 | [PyTorch device调优](#pytorch-device调优) |
| **训练** | Device内存瓶颈 | 显存不足、OOM | [PyTorch device调优](#pytorch-device调优) |
| **推理** | 多实例资源抢占 | 多实例部署时性能波动 | [吞吐优化](#吞吐优化) |
| **推理** | 多Stream并发卡死 | 并发场景出现卡死/死锁 | [吞吐优化](#吞吐优化) |
| **通用** | 需采集性能数据 | 尚无性能数据，需先采集 | [数据采集](#数据采集) |
| **通用** | 需可视化分析 | 已有数据，需可视化定位瓶颈 | [性能可视化](#性能可视化) |
| **通用** | 需性能比对 | GPU vs NPU 或 NPU vs NPU 对比 | [性能比对](#性能比对可选) |
| **通用** | 需参考实操案例 | 想看端到端调优样例 | [性能数据分析样例](#性能数据分析样例) |

## 性能调优流程

调优前请先根据使用的训练框架参考相应的调优流程，不同的框架调优手段、工具存在差异。
Rec SDK适配了PyTorch、TensorFlow，可以参考以下AI框架层的调优流程：

* [PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/performance_tuning_process.md)：介绍PyTorch模型在昇腾设备上的端到端性能调优流程，包括数据采集、瓶颈定位、优化实施各阶段。
* [TensorFlow](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000039.html)：介绍TensorFlow模型在昇腾设备上的性能调优流程，涵盖训练性能分析和常见优化手段。

## 工具介绍

调优流程通常涉及性能数据采集、数据可视化用于识别性能瓶颈点。
本章节列举常用工具供用户参考使用。

### 数据采集

在模型代码中通过框架层接口调用，可获取框架层算子信息、CANN层算子信息、底层NPU算子信息以及算子内存占用信息等。基于工具获取的数据可用于分析算子性能，识别host、device、通信、IO瓶颈：

* [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0033.html)：PyTorch框架的性能数据采集接口，支持采集算子耗时、通信耗时、内存占用等多维度数据。
* [TensorFlow Profiling API](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0037.html)：TensorFlow框架的性能数据采集接口，支持采集算子级、图级性能数据用于瓶颈分析。

### 性能可视化

MindStudio Insight是面向昇腾AI开发者的可视化调优工具，能够可视化呈现真实软硬件运行数据，多维度分析性能瓶颈点，支持百卡、千卡及以上规模的可视化集群性能分析。
工具提供时间线视图、内存、算子耗时、通信瓶颈分析等功能，帮助开发者快速定位模型性能瓶颈：

* [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md?framework=mindspore)：可视化调优工具，提供时间线视图、算子耗时排名、通信瓶颈分析、内存分析等功能，支持大规模集群性能分析。

### 性能比对（可选）

compare_tools支持比较GPU与NPU之间、NPU与NPU之间的性能差异，通过对训练耗时和内存占用的比对分析，定位到具体劣化的算子，帮助用户提升性能调优的效率。工具将训练耗时拆分为计算、通信、调度三大维度，并针对计算和通信分别进行算子级别的比对；将训练占用的总内存，拆分成算子级别的内存占用进行比对。

* [compare_tools](https://gitcode.com/Ascend/mstt/tree/master/profiler/msprof_analyze/compare_tools)：性能比对工具，支持GPU与NPU、NPU与NPU之间的耗时和内存比对，从计算、通信、调度三维度定位劣化算子。

## 性能数据分析样例

本章节列举的案例包含工具应用、数据分析、调优的具体流程，仅供参考：

* [MindStudio Insight样例](https://www.hiascend.com/document/detail/zh/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md)：使用MindStudio Insight进行性能问题定位的端到端案例，涵盖常见性能问题的排查思路和操作步骤。
* [Rec SDK样例](./02_performance_tuning.md)：Rec SDK场景下的性能调优实操案例，包含性能数据采集、瓶颈分析、优化实施完整流程。

## 性能调优方法

调优方法主要分为host、device两大类，部分方法应用与AI框架相关，建议用户优先参考以下调优策略，识别具体调优场景，再选择合适的优化方法进行应用。

### 总体策略

* [通用调优策略](https://www.hiascend.com/document/detail/zh/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md#排查思路介绍)：适用于所有框架的性能问题排查思路，帮助系统化定位瓶颈所在。

* [PyTorch调优策略](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/overall_optimization_strategy.md)：PyTorch模型在昇腾上的整体优化策略，按优先级梳理各阶段调优方向。

### Host优化

#### 通用方法

* [CPU自动绑核](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000053.html)：将训练进程绑定到指定CPU核心，减少跨NUMA节点访问和线程迁移开销，提升host侧执行效率。
* OS优化
  * [使用高性能内存库](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hi_perf_mem_pool_sub.md)：替换系统默认内存分配器为高性能内存池，降低内存分配和释放的开销。
  * [大页内存](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hp_opt.md)：启用大页内存减少TLB miss，提升内存访问效率，适用于大内存占用的训练场景。

#### PyTorch host调优

* [PyTorch绑核优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/core_bind_opt.md)：避免线程间抢占，提高缓存命中率，避免跨NUMA（非统一内存访问架构）节点的内存访问，减少任务调度开销，优化任务执行效率。
* [算子下发流水优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/pipeline_opt.md)：优化host侧算子下发流程，使算子下发与device执行并行，减少device空闲等待时间。
* [编译优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/comp_opt_intro.md)：通过算子编译缓存、图编译优化等手段缩短模型编译耗时，加速训练启动和迭代执行。

### Device优化

#### 通信优化

* [通信概述和优化方法](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/communication_basics_overview.md)：介绍昇腾设备上的通信原理、常见通信算子及优化手段，帮助降低多卡训练的通信开销。

#### 吞吐优化

为充分利用硬件空闲资源部署多个推理实例时，易出现的硬件资源竞争问题，通过控制核心资源分配可避免实例间资源抢占冲突，保障各实例稳定运行，有效提升吞吐量并降低时延。

* [限制算子执行核心数](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-set_device_limit.md)：通过限制单个推理实例可使用的AICore数量，避免多实例间资源抢占，保障各实例稳定运行。
* [限制device资源](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1879.html)：通过ACL接口限制device可用计算资源，适用于多实例推理场景的资源隔离。
* [调优案例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/graphdevg/atlasag_25_0101.html)：多实例推理场景下的吞吐优化案例，展示如何通过控核实现资源分配和性能提升。

此外，多Stream并发场景可能会出现卡死现象，同样需要限制核心资源进行规避处理。

* [多Stream并发场景控核](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/appendix/appendix/core_limit.md)：解决多Stream并发场景下的卡死问题，通过限制核心资源分配规避死锁。

#### PyTorch device调优

* [数据IO优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/data_loading_optimization.md)：优化数据加载和预处理流程，减少NPU等待数据的时间，提升数据供给效率。
* 融合算子替换
  * [RotaryMul & RotaryMulGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/RotaryMul-RotaryMulGrad.md)：将旋转位置编码的乘法与梯度计算融合为单个算子，减少kernel launch开销和显存访问次数。
  * [RmsNorm & RmsNormGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/RmsNorm-RmsNormGrad.md)：将RMS归一化及其梯度计算融合，减少中间结果的显存读写，提升LLM模型训练效率。
  * [ScaledMaskedSoftmax & ScaledMaskedSoftmaxGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/ScaledMaskedSoftmax-ScaledMaskedSoftmaxGrad.md)：将缩放掩码Softmax及其梯度融合，适用于注意力机制中的掩码计算场景。
  * [MatmulAllReduce](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/MatmulAllReduce.md)：将矩阵乘与AllReduce通信融合，减少中间结果回落host的开销，加速分布式训练。
  * [FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/FlashAttentionScore.md)：高性能注意力计算融合算子，显著降低显存占用和计算耗时，适用于大模型训练。
  * [SwiGlu](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/SwiGlu.md)：将Swish激活与GLU门控融合为单个算子，减少FFN层的显存访问和kernel开销。
* [融合优化器替换](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/fusion_opt.md)：将优化器更新步骤融合为单个算子，减少小算子数量和显存访问，加速反向传播后的参数更新。
* 亲和算子替换
  * [矩阵索引 IndexPut](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/indexput_operator_replacement.md)：用昇腾亲和实现替换原生IndexPut算子，解决原生算子在NPU上性能较差的问题。
  * [Nonzero](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/nonzero_op_replacement.md)：用昇腾亲和实现替换原生Nonzero算子，提升非零元素索引的计算效率。
  * [where](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/where_op_replace.md)：用昇腾亲和实现替换原生where算子，提升条件选择操作的计算效率。
* [内存优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/optimization_recommendations.md)：通过显存复用、梯度累积、内存池优化等手段降低显存占用，避免OOM并提升训练吞吐。

#### TensorFlow device调优

* [混合精度训练](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000042.html)：使用float16与float32混合计算，在保持精度的同时利用NPU的半算力优势提升训练速度。
* [亲和算子替换](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000046.html)：用昇腾亲和算子替换TensorFlow原生算子，解决部分算子在NPU上性能不佳的问题。
* [训练迭代循环下沉](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000048.html)：将训练迭代循环下沉到device侧执行，减少host与device之间的交互次数，提升迭代效率。
* [AOE自动调优](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000052.html)：自动调优引擎，根据硬件特征自动优化算子编译策略，无需手动调参即可提升性能。
