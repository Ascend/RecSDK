# Performance Optimization Overview

This document provides an overview of the full Ascend performance tuning process, along with the corresponding tools and methods for performance data collection, analysis, and tuning.

## Performance Tuning Process

Before tuning, first refer to the corresponding tuning process based on the training framework you use. Different frameworks use different tuning methods and tools.
Rec SDK supports PyTorch and TensorFlow. You can refer to the following tuning processes at the AI framework layer:

* [PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/performance_tuning_process.md)
* [TensorFlow](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000039.html)

## Tool Introduction

The tuning process usually involves performance data collection and data visualization to identify performance bottlenecks.
This section lists commonly used tools for reference.

### Data Collection

By calling framework-layer interfaces in the model code, you can obtain framework-layer operator information, CANN-layer operator information, underlying NPU operator information, and operator memory usage information. The data obtained by these tools can help analyze operator performance and identify host, device, communication, and I/O bottlenecks:

* [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0033.html)
* [TensorFlow Profiling API](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0037.html)

### Performance Visualization

MindStudio Insight is a visual tuning tool for Ascend AI developers. It visualizes real hardware and software runtime data, analyzes performance bottlenecks from multiple dimensions, and supports visual cluster performance analysis at scales from hundreds to thousands of cards and beyond.
The tool provides timeline views, memory analysis, operator duration analysis, communication bottleneck analysis, and other functions to help developers quickly locate model performance bottlenecks:

* [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md?framework=mindspore)

### Performance Comparison (Optional)

compare_tools supports comparing performance differences between GPU and NPU, and between different NPUs. By comparing training duration and memory usage, it pinpoints specific degraded operators and helps improve tuning efficiency. The tool breaks training duration down into three dimensions: computation, communication, and scheduling. It performs operator-level comparisons for computation and communication separately. It also breaks down the total memory used for training into operator-level memory usage for comparison.

* [compare_tools](https://gitcode.com/Ascend/mstt/tree/master/profiler/msprof_analyze/compare_tools)

## Performance Data Analysis Examples

The examples listed in this section include the specific process of tool usage, data analysis, and tuning. They are for reference only:

* [MindStudio Insight example](https://www.hiascend.com/document/detail/zh/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md)
* [Rec SDK example](./performance_tuning.md)

## Performance Tuning Methods

Tuning methods are mainly divided into host-side and device-side methods. Some methods are framework-specific. You are advised to first refer to the following tuning strategies, identify the specific tuning scenario, and then choose the appropriate optimization method to apply.

### Overall Strategy

* [General tuning strategy](https://www.hiascend.com/document/detail/zh/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md#排查思路介绍)

* [PyTorch tuning strategy](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/overall_optimization_strategy.md)

### Host Optimization

#### General Methods

* [Automatic CPU core binding](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000053.html)
* OS optimization
  * [Use a high-performance memory library](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hi_perf_mem_pool_sub.md)
  * [Large page memory](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hp_opt.md)

#### PyTorch Host Optimization

* [PyTorch core binding optimization](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/core_bind_opt.md) Avoid thread preemption, improve cache hit rates, avoid memory access across NUMA (non-uniform memory access architecture) nodes, reduce task scheduling overhead, and improve task execution efficiency.
* [Operator dispatch pipeline optimization](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/pipeline_opt.md)
* [Compilation optimization](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/comp_opt_intro.md)

### Device Optimization

#### Communication Optimization

* [Communication overview and optimization methods](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/communication_basics_overview.md)

#### PyTorch Device Optimization

* [Data I/O optimization](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/data_loading_optimization.md)
* Fused operator replacement
  * [RotaryMul & RotaryMulGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/RotaryMul-RotaryMulGrad.md)
  * [RmsNorm & RmsNormGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/RmsNorm-RmsNormGrad.md)
  * [ScaledMaskedSoftmax & ScaledMaskedSoftmaxGrad](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/ScaledMaskedSoftmax-ScaledMaskedSoftmaxGrad.md)
  * [MatmulAllReduce](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/MatmulAllReduce.md)
  * [FlashAttentionScore](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/FlashAttentionScore.md)
  * [SwiGlu](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/SwiGlu.md)
* [Fused optimizer replacement](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/fusion_opt.md)
* Affinity operator replacement
  * [Matrix indexing IndexPut](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/indexput_operator_replacement.md)
  * [Nonzero](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/nonzero_op_replacement.md)
  * [where](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/where_op_replace.md)
* [Memory optimization](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/optimization_recommendations.md)

#### TensorFlow Device Optimization

* [Mixed-precision training](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000042.html)
* [Affinity operator replacement](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000046.html)
* [Training iteration loop sinking](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000048.html)
* [AOE automatic tuning](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000052.html)
