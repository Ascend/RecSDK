# AsynchronousCompleteCumsum算子及样例说明
本算子仅支持NPU调用

## AsynchronousCompleteCumsum算子文件结构

```shell
├── asynchronous_complete_cumsum.json    # 算子原型配置
├── op_host    # AsynchronousCompleteCumsum算子Host侧实现
├── op_kernel  # AsynchronousCompleteCumsum算子Kernel侧实现
├── README.md  # AsynchronousCompleteCumsum算子说明文档
└── run.sh     # AsynchronousCompleteCumsum算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## AsynchronousCompleteCumsum算子使用

1. 上传asynchronous_complete_cumsum文件夹到目标环境，并进入当前目录，执行指令对asynchronous_complete_cumsum算子进行编译和部署

默认编译安装Atlas A5训练系列产品AI Core类型：
```shell
bash run.sh
```

指定 AI Core 类型编译，目前此版本支持Atlas A5系列产品：

```shell
bash run.sh ai_core-<soc_version>
```
> AI处理器的型号<soc_version>请通过如下方式获取:
> - 在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，获取`Chip Name`信息。实际配置值为AscendChip Name，例如`Chip Name`取值为`xxxyy`，实际配置值为`Ascendxxxyy`。
>
> 基于同系列的AI处理器型号创建的算子工程，其基础功能（基于该工程进行算子开发、编译和部署）通用。

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。安装8.3.RC1版本的CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/8.3.RC1/bin/setenv.bash
export ASCEND_OPP_KERNEL_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## asynchronous_complete_cumsum算子介绍

1. 算子分析

a) 算子的主要功能是实现输入offset的异步并行累加（前缀和计算）；  
b) 算子输入说明：
* x：输入的offset tensor, eg: [1,5,6]

c) 算子输出说明：
* y：输入的offset tensor对应的累加和, eg: [0, 1, 6, 12]
  - 前N个元素为前缀和结果：[0, 1, 6]
  - 最后一个元素为总和：12

d) 算法特点：
* **异步并行**：支持多AI Core并行计算，充分利用硬件资源
* **双模式优化**：根据数据规模自动选择最优算法策略
* **内存友好**：使用Cache对齐和共享内存优化，减少全局内存访问
* **负载均衡**：智能分配Block到不同AI Core，确保计算负载均衡

e) 算子约束说明：
* 支持的型号：Atlas A5系列产品;
* 支持的CANN版本：8.3.RC1及之后版本；
* 支持的输入数据类型：int32；int64
* 输入的数据只支持1维。
* 算子参数均会在NPU显存中存放，请根据显存大小合理设置参数长度。
* 最大支持输入长度：理论上无限制，实际受NPU显存大小约束

2. Host侧算子实现

Host侧算子实现在目录 op_host下，包括asynchronous_complete_cumsum.cpp和asynchronous_complete_cumsum_tiling.h。

a) Tiling实现

**TilingFunc函数**：
- 从context中获取输入参数信息（输入shape、数据类型等）
- 进行输入有效性校验：确保输入为1维张量，支持int32/int64数据类型
- 根据输入长度选择算法策略
- 计算资源分配：
  - 根据输入长度计算所需Block数量
  - 获取可用AI Core数量，实现负载均衡
  - 计算每个Core处理的Block数量
- 工作空间计算：
   - SharedMemory空间：存储每个Block的累积和
  - 系统工作空间：AscendC平台所需空间

**TilingData结构**：
- totalLength：输入数据总长度
- totalBlocks：总Block数量
- blocksPerCore：每个Core处理的Block数量
- remainderBlocks：余数Block数量（用于负载均衡）
- elementsPerBlock：单线程块处理的最大元素个数
- isSmall：判断是否是小数据
- isFullCore：判断是否用到全部的核

b) Shape推导

**InferShape函数**：
- 输入：1维张量，长度为N
- 输出：1维张量，长度为N+1
- 前N个元素为前缀和结果，第N+1个元素为总和
- 支持动态shape（-1表示未知长度）

**InferDataType函数**：
- 输出数据类型与输入数据类型保持一致
- 支持int32和int64两种数据类型

c) 原型注册

**AsynchronousCompleteCumsum类**：
- 继承自OpDef基类
- 定义输入输出参数：
  - 输入"x"：必需参数，支持int32/int64，ND格式
  - 输出"y"：必需参数，支持int32/int64，ND格式
- 绑定推理函数：InferShape和InferDataType
- 绑定Tiling函数：optiling::TilingFunc
- 配置AI Core支持：ascend950

3. Kernel侧算子实现

Kernel侧算子实现在目录op_kernel下，其中包括：asynchronous_complete_cumsum.cpp和asynchronous_complete_cumsum_kernel.h。

a) 核函数的入口：`extern "C" __global__ __aicore__ void asynchronous_complete_cumsum`

b) 解析tiling参数：`GET_TILING_DATA(tilingData, tiling)`从TilingData中获取host侧传入的数据

c) 核心算法实现：

   **小数据模式**：
   - 使用SIMT向量化函数`SimtSmallDataCompute`进行并行计算
   - 采用Warp级前缀和算法：每个Warp内部使用`WarpPrefixSum`进行并行前缀和计算
   - 通过共享内存实现Block级同步和前缀和聚合
   - 支持多Block场景下的两阶段更新：使用ReduceSum指令计算块级的前缀和

   **大数据模式**：
   - 第一阶段：`SimtLargeDataCompute` - 每个线程处理多个元素，计算局部前缀和
   - 第二阶段：使用ReduceSum指令计算块级的前缀和，更新当前Block的输出
   - 支持多核并行：每个AI Core处理多个Block，通过`blocksPerCore`和`remainderBlocks`实现负载均衡

d) 关键技术特性：

   **内存优化**：
  - 共享内存用于Block内Warp间通信，减少全局内存访问

   **并行策略**：
   - 最大支持1024个线程/Block，32个线程/Warp
   - 小数据模式：每个线程处理1个元素
   - 大数据模式：每个线程最多处理4个元素
   - 支持多AI Core并行执行

   **前缀和算法**：
   - Warp级前缀和：使用`WarpShflUpSync`进行线程间数据交换
   - Block级前缀和：通过共享内存聚合Warp结果
   - 全局前缀和：通过BlockSums实现跨Block的累积
   
4. 性能优化特性

a) **内存访问优化**：
   - 共享内存通信：Block内Warp间通过共享内存通信，减少全局内存访问
   - 向量化处理：使用SIMT向量化函数，提高指令吞吐量

b) **并行计算优化**：
   - 多级并行：Warp级、Block级、Core级三级并行
   - 负载均衡：智能分配Block到不同AI Core，避免负载不均
   - 异步执行：支持多Block异步并行执行
   
