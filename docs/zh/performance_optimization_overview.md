# 性能优化总览

本文档汇集昇腾性能调优全流程以及相应的工具、方法，供开发者进行性能数据采集、分析、调优参考。

## 性能调优流程

调优前请先根据使用的训练框架参考相应的调优流程，不同的框架调优手段、工具存在差异。
Rec SDK适配了PyTorch、TensorFlow，可以参考以下AI框架层的调优流程：

* [PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/performance_tuning_process.md)
* [TensorFlow](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000039.html)

## 工具介绍

调优流程通常涉及性能数据采集、数据可视化用于识别性能瓶颈点。
本章节列举常用工具供用户参考使用。

### 数据采集

在模型代码中通过框架层接口调用，可获取框架层算子信息、CANN层算子信息、底层NPU算子信息以及算子内存占用信息等。基于工具获取的数据可用于分析算子性能，识别host、device、通信、IO瓶颈：

* [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0033.html)
* [TensorFlow Profiling API](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0037.html)

### 性能可视化

MindStudio Insight是面向昇腾AI开发者的可视化调优工具，能够可视化呈现真实软硬件运行数据，多维度分析性能瓶颈点，支持百卡、千卡及以上规模的可视化集群性能分析。
工具提供时间线视图、内存、算子耗时、通信瓶颈分析等功能，帮助开发者快速定位模型性能瓶颈：

* [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md?framework=mindspore)

### 性能比对（可选）

compare_tools支持比较GPU与NPU之间、NPU与NPU之间的性能差异，通过对训练耗时和内存占用的比对分析，定位到具体劣化的算子，帮助用户提升性能调优的效率。工具将训练耗时拆分为计算、通信、调度三大维度，并针对计算和通信分别进行算子级别的比对；将训练占用的总内存，拆分成算子级别的内存占用进行比对。

* [compare_tools](https://gitcode.com/Ascend/mstt/tree/master/profiler/msprof_analyze/compare_tools)

## 性能数据分析样例

本章节列举的案例包含工具应用、数据分析、调优的具体流程，仅供参考：

* [MindStudio Insight样例](https://www.hiascend.com/document/detail/zh/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md)
* [Rec SDK样例](./performance_tuning.md)

## 性能调优方法

调优方法主要分为host、device两大类，部分方法应用与AI框架相关，建议用户优先参考以下调优策略，识别具体调优场景，再选择合适的优化方法进行应用。

### 总体策略

* [通用调优策略](https://www.hiascend.com/document/detail/zh/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md#排查思路介绍 )

* [PyTorch调优策略](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/overall_optimization_strategy.md)

### Host优化

#### 通用方法

* [CPU自动绑核](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000053.html)
* OS优化
  * [使用高性能内存库](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hi_perf_mem_pool_sub.md)
  * [大页内存](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hp_opt.md)

#### PyTorch host调优

* [PyTorch绑核优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/core_bind_opt.md) 避免线程间抢占，提高缓存命中率，避免跨NUMA（非统一内存访问架构）节点的内存访问，减少任务调度开销，优化任务执行效率。
* [算子下发流水优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/pipeline_opt.md)
* [编译优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/comp_opt_intro.md)

### Device优化

#### 通信优化

* [通信概述和优化方法](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/communication_basics_overview.md)

#### PyTorch device调优

* [数据IO优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/data_loading_optimization.md)
* 融合算子替换
  * [RotaryMul & RotaryMulGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/RotaryMul-RotaryMulGrad.md)
  * [RmsNorm & RmsNormGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/RmsNorm-RmsNormGrad.md)
  * [ScaledMaskedSoftmax & ScaledMaskedSoftmaxGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/ScaledMaskedSoftmax-ScaledMaskedSoftmaxGrad.md)
  * [MatmulAllReduce](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/MatmulAllReduce.md)
  * [FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/FlashAttentionScore.md)
  * [SwiGlu](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/SwiGlu.md)
* [融合优化器替换](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/fusion_opt.md)
* 亲和算子替换
  * [矩阵索引 IndexPut](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/indexput_operator_replacement.md)
  * [Nonzero](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/nonzero_op_replacement.md)
  * [where](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/where_op_replace.md)
* [内存优化](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/optimization_recommendations.md)

#### TensorFlow device调优

* [混合精度训练](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000042.html)
* [亲和算子替换](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000046.html)
* [训练迭代循环下沉](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000048.html)
* [AOE自动调优](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000052.html)
