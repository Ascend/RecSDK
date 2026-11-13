# Performance Optimization Overview

This document provides an overview of the full Ascend performance tuning process, along with the corresponding tools and methods for performance data collection, analysis, and tuning.

## Quick Navigation: Scenario Decision Table

Based on your scenario and bottleneck type, quickly locate the corresponding section:

| Scenario | Bottleneck Type | Symptom | Jumping to Section |
|---------|---------|---------|---------|
| **Training** | Uncertain bottleneck location | Overall performance does not meet expectations. Systematic troubleshooting is required | [Overall strategy](#overall-strategy) → [Performance tuning process](#performance-tuning-process) |
| **Training** | Host CPU bottleneck | High CPU utilization and large operator dispatch latency | [Host optimization - General methods](#general-methods) → [PyTorch host optimization](#pytorch-host-optimization) |
| **Training** | Host memory bottleneck | High memory allocation time and frequent page replacement | [Host optimization - General methods](#general-methods) |
| **Training** | Host compilation bottleneck | Long model compilation time and slow first iteration | [PyTorch host optimization](#pytorch-host-optimization) |
| **Training** | Device compute bottleneck | Low NPU utilization and concentrated operator duration | [PyTorch device optimization](#pytorch-device-optimization)/[TensorFlow device optimization](#tensorflow-device-optimization) |
| **Training** | Device communication bottleneck | High communication time ratio and low multi-device scaling efficiency | [Communication optimization](#communication-optimization) |
| **Training** | Device I/O bottleneck | High data loading time and NPU waiting for data | [PyTorch device optimization](#pytorch-device-optimization) |
| **Training** | Device memory bottleneck | Insufficient memory and OOM | [PyTorch device optimization](#pytorch-device-optimization) |
| **Inference** | Multi-instance resource contention | Performance fluctuation in multi-instance deployment | [Throughput optimization](#throughput-optimization) |
| **Inference** | Multi-Stream concurrency deadlock | Deadlock or hang in concurrent scenarios | [Throughput optimization](#throughput-optimization) |
| **General** | Need to collect performance data | No performance data is available yet. Collect it first | [Data collection](#data-collection) |
| **General** | Need visual analysis | Data is available. Use visualization to locate bottlenecks | [Performance visualization](#performance-visualization) |
| **General** | Need performance comparison | GPU vs NPU or NPU vs NPU comparison | [Performance comparison](#performance-comparison-optional) |
| **General** | Need reference use cases | Want to see end-to-end tuning examples | [Performance data analysis examples](#performance-data-analysis-examples) |

## Performance Tuning Process

Before tuning, first refer to the corresponding tuning process based on the training framework you use. Different frameworks use different tuning methods and tools.
Rec SDK supports PyTorch and TensorFlow. You can refer to the following tuning processes at the AI framework layer:

* [PyTorch](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/performance_tuning_process.md): Introduces the end-to-end performance tuning process for PyTorch models on Ascend devices, covering data collection, bottleneck identification, and optimization implementation.
* [TensorFlow](https://www.hiascend.com/document/detail/en/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000039.html): Introduces the performance tuning process for TensorFlow models on Ascend devices, covering training performance analysis and common optimization methods.

## Tool Introduction

The tuning process usually involves performance data collection and data visualization to identify performance bottlenecks.
This section lists commonly used tools for reference.

### Data Collection

By calling framework-layer interfaces in the model code, you can obtain framework-layer operator information, CANN-layer operator information, underlying NPU operator information, and operator memory usage information. The data obtained by these tools can help analyze operator performance and identify host, device, communication, and I/O bottlenecks:

* [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/canncommercial/900/devaids/Profiling/atlasprofiling_16_0033.html): The performance data collection interface of the PyTorch framework. It supports collecting multi-dimensional data such as operator duration, communication duration, and memory usage.
* [TensorFlow Profiling API](https://www.hiascend.com/document/detail/en/canncommercial/900/devaids/Profiling/atlasprofiling_16_0037.html): The performance data collection interface of the TensorFlow framework. It supports collecting operator-level and graph-level performance data for bottleneck analysis.

### Performance Visualization

MindStudio Insight is a visual tuning tool for Ascend AI developers. It visualizes real hardware and software runtime data, analyzes performance bottlenecks from multiple dimensions, and supports visual cluster performance analysis at scales from hundreds to thousands of devices and beyond.
The tool provides timeline views, memory analysis, operator duration analysis, communication bottleneck analysis, and other functions to help developers quickly locate model performance bottlenecks:

* [MindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/en/user_guide/overview.md?framework=mindspore): A visual tuning tool that provides timeline views, operator duration ranking, communication bottleneck analysis, memory analysis, and other functions, and supports performance analysis of large-scale clusters.

### Performance Comparison (Optional)

`compare_tools` supports comparing performance differences between GPU and NPU, and between different NPUs. By comparing training duration and memory usage, it pinpoints specific degraded operators and helps improve tuning efficiency. The tool breaks training duration down into three dimensions: computation, communication, and scheduling. It performs operator-level comparisons for computation and communication separately. It also breaks down the total memory used for training into operator-level memory usage for comparison.

* `compare_tools`: A performance comparison tool that supports duration and memory comparison between GPU and NPU and between different NPUs, and locates degraded operators from the three dimensions of computation, communication, and scheduling. Install guide: compare_tools is integrated in msprof-analyze. You need to install [msprof-analyze](https://gitcode.com/Ascend/msprof-analyze/blob/26.1.0/README_EN.md#-installation-guide). Usage guide: [compare_tool_instruct](https://gitcode.com/Ascend/msprof-analyze/blob/26.1.0/docs/en/user_guide/compare_tool_instruct.md).

## Performance Data Analysis Examples

The examples listed in this section include the specific process of tool usage, data analysis, and tuning. They are for reference only:

* [MindStudio Insight example](https://www.hiascend.com/document/detail/en/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md): An end-to-end case of using MindStudio Insight to locate performance issues, covering the troubleshooting ideas and operating steps for common performance problems.
* [Rec SDK example](./performance_tuning.md) A hands-on performance tuning case in Rec SDK scenarios, covering the complete process of performance data collection, bottleneck analysis, and optimization implementation.

## Performance Tuning Methods

Tuning methods are mainly divided into host-side and device-side methods. Some methods are framework-specific. You are advised to first refer to the following tuning strategies, identify the specific tuning scenario, and then choose the appropriate optimization method to apply.

### Overall Strategy

* [General tuning strategy](https://www.hiascend.com/document/detail/en/mindstudio/2600/practicalcases/GeneralPerformanceIssue/MindStudio/26.0.0/cases/general_performance_issue_troubleshooting_guide/positioning_process_for_performance_issues.md): Troubleshooting ideas applicable to all frameworks, helping you systematically locate bottlenecks.

* [PyTorch tuning strategy](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/overall_optimization_strategy.md): The overall optimization strategy for PyTorch models on Ascend, organizing the tuning directions of each stage by priority.

### Host Optimization

#### General Methods

* [Automatic CPU core binding](https://www.hiascend.com/document/detail/en/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000053.html): Binds the training process to specified CPU cores to reduce cross-NUMA-node access and thread migration overhead, improving host-side execution efficiency.
* OS optimization
  * [Use a high-performance memory library](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/hi_perf_mem_pool_sub.md): Replaces the system default memory allocator with a high-performance memory pool to reduce memory allocation and release overhead.
  * [Large page memory](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/hp_opt.md): Enables large page memory to reduce TLB misses and improve memory access efficiency. It is suitable for training scenarios with large memory usage.

#### PyTorch Host Optimization

* [PyTorch core binding optimization](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/core_bind_opt.md): Avoid thread preemption, improve cache hit rates, avoid memory access across non-uniform memory access architecture (NUMA) nodes, reduce task scheduling overhead, and improve task execution efficiency.
* [Operator dispatch pipeline optimization](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/pipeline_opt.md): Optimizes the host-side operator dispatch process so that operator dispatch runs in parallel with device execution, reducing device idle wait time.
* [Compilation optimization](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/comp_opt_intro.md): Shortens model compilation time through operator compilation caching, graph compilation optimization, and other methods, accelerating training startup and iteration execution.

### Device Optimization

#### Communication Optimization

* [Communication overview and optimization methods](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/communication_basics_overview.md): Introduces the communication principles, common communication operators, and optimization methods on Ascend devices, helping reduce communication overhead in multi-device training.

#### Throughput Optimization

When multiple inference instances are deployed to fully utilize idle hardware resources, hardware resource contention may occur. By controlling core resource allocation, you can avoid resource preemption conflicts between instances, ensure stable operation of each instance, and effectively improve throughput while reducing latency.

* [Limit the number of operator execution cores](https://www.hiascend.com/document/detail/en/Pytorch/2600/apiref/torchnpuCustomsapi/docs/en/custom_APIs/torch_npu/torch_npu-set_device_limit.md): Limits the number of AICores available to a single inference instance to avoid resource preemption among multiple instances and ensure stable operation of each instance.
* [Limit device resources](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1879.html): Limits the computing resources available to a device through ACL interfaces. It is suitable for resource isolation in multi-instance inference scenarios.
* [Tuning case](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/graphdevg/atlasag_25_0101.html): A throughput optimization case in multi-instance inference scenarios, showing how to achieve resource allocation and performance improvement through core limiting.

In addition, deadlock may occur in multi-Stream concurrent scenarios. You also need to limit core resources to avoid this problem.

* [Core limiting in multi-Stream concurrent scenarios](https://www.hiascend.com/document/detail/en/Pytorch/2600/modthirdparty/torchairuseguide/docs/en/appendix/appendix/core_limit.md): Resolves the deadlock problem in multi-Stream concurrent scenarios by limiting core resource allocation to avoid deadlocks.

#### PyTorch Device Optimization

* [Data I/O optimization](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/data_loading_optimization.md): Optimizes the data loading and preprocessing process to reduce the time the NPU waits for data and improve data supply efficiency.
* Fused operator replacement
  * [RotaryMul & RotaryMulGrad](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/RotaryMul-RotaryMulGrad.md): Fuses the multiplication of rotary position encoding and its gradient computation into a single operator, reducing kernel launch overhead and the number of memory accesses.
  * [RmsNorm & RmsNormGrad](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/RmsNorm-RmsNormGrad.md): Fuses RMS normalization and its gradient computation to reduce memory read/write of intermediate results, improving LLM model training efficiency.
  * [ScaledMaskedSoftmax & ScaledMaskedSoftmaxGrad](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/ScaledMaskedSoftmax-ScaledMaskedSoftmaxGrad.md): Fuses scaled masked Softmax and its gradient, applicable to mask computation scenarios in attention mechanisms.
  * [MatmulAllReduce](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/MatmulAllReduce.md): Fuses matrix multiplication with AllReduce communication to reduce the overhead of intermediate results falling back to the host, accelerating distributed training.
  * [FlashAttentionScore](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/FlashAttentionScore.md): A high-performance fused attention computation operator that significantly reduces memory usage and computation time, suitable for LLM training.
  * [SwiGlu](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/SwiGlu.md): Fuses the Swish activation and GLU gating into a single operator, reducing memory access and kernel overhead in FFN layers.
* [Fused optimizer replacement](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/fusion_opt.md): Fuses optimizer update steps into a single operator, reducing the number of small operators and memory access, accelerating parameter updates after backpropagation.
* Affinity operator replacement
  * [Matrix indexing IndexPut](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/indexput_operator_replacement.md): Replaces the native IndexPut operator with an Ascend affinity implementation to solve the poor performance of the native operator on NPUs.
  * [Nonzero](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/nonzero_op_replacement.md): Replaces the native Nonzero operator with an Ascend affinity implementation to improve the computation efficiency of nonzero element indexing.
  * [where](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/where_op_replace.md): Replaces the native where operator with an Ascend affinity implementation to improve the computation efficiency of conditional selection operations.
* [Memory optimization](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/optimization_recommendations.md): Reduces memory usage through memory reuse, gradient accumulation, memory pool optimization, and other methods, avoiding OOM and improving training throughput.

#### TensorFlow Device Optimization

* [Mixed-precision training](https://www.hiascend.com/document/detail/en/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000042.html): Uses a mix of `float16` and `float32` computation to leverage the half-precision compute advantage of NPUs and improve training speed while maintaining accuracy.
* [Affinity operator replacement](https://www.hiascend.com/document/detail/en/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000046.html): Replaces native TensorFlow operators with Ascend affinity operators to solve the poor performance of some operators on NPUs.
* [Training iteration loop sinking](https://www.hiascend.com/document/detail/en/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000048.html): Sinks the training iteration loop to the device side for execution, reducing the number of interactions between the host and device and improving iteration efficiency.
* [AOE automatic tuning](https://www.hiascend.com/document/detail/en/TensorFlowCommercial/900/migration/tfmigr1/tfmigr1_000052.html): An automatic tuning engine that automatically optimizes operator compilation strategies based on hardware characteristics, improving performance without manual parameter tuning.
