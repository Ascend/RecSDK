# InvertPermute算子及样例说明
本算子仅支持NPU调用

## InvertPermute算子文件结构

```shell
├── InvertPermute.json    # 算子原型配置
├── op_host    # Host侧实现
├── op_kernel  # Kernel侧实现
├── README.md  # 说明文档
└── run.sh     # 安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## InvertPermute算子使用

1. 上传invert_permute文件夹到目标环境，并进入当前目录，执行指令对invert_permute算子进行编译和部署

默认编译安装Atlas A5训练系列产品AI Core类型：
```shell
bash run.sh
```

指定 AI Core 类型编译，目前此版本只支持Atlas A5系列产品：

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

## invert_permute算子介绍

1. 算子分析

a) 算子的主要功能是对输入张量进行逆置换操作；  
b) 算子输入说明：
* x：输入张量, eg: tensor([2,0,1])

c) 算子输出说明：
* y：输入张量的逆序列, eg: tensor([1,2,0])

d) 算子约束说明：
* 支持的型号：Atlas A5系列产品;
* 支持的CANN版本：8.3.RC1及之后版本；
* 支持的输入数据类型：int32、int64；
* 输入的数据需为permute数据，满足合法一维排列的全部数学特征；

2. Host侧算子实现

Host侧算子实现在目录 op_host下，包括invert_permute.cpp和invert_permute_tiling.h。

a) Tiling实现

**TilingFunc函数**：
- 从context中获取输入参数信息（输入shape、数据类型等）
- 进行输入有效性校验：确保输入为1维张量，支持int32/int64数据类型
- 根据输入长度选择算法策略
- 计算资源分配：
  - 根据输入长度计算所需Block数量以及线程数
  - 获取可用AI Core数量，实现负载均衡

**TilingData结构**：
- xDim0：输入数据元素个数
- actualThreadsPerBlock：单Block实际使用的线程数

b) Shape推导

**InferShape函数**：
- 输入：1维张量，长度为N
- 输出：1维张量，长度为N

**InferDataType函数**：
- 输出的数据类型、shape与输入保持一致
- 支持int32和int64两种数据类型

c) 原型注册

**InvertPermute类**：
- 继承自OpDef基类
- 定义输入输出参数：
  - 输入"x"：必需参数，支持int32/int64，ND格式
  - 输出"y"：必需参数，支持int32/int64，ND格式
- 绑定推理函数：InferShape和InferDataType
- 绑定Tiling函数：optiling::TilingFunc
- 配置AI Core支持：ascend910_95

3. Kernel侧算子实现

Kernel侧算子实现在目录op_kernel下，其中包括：invert_permute.cpp和invert_permute_kernel.h。

a) 核函数的入口：`extern "C" __global__ __aicore__ void invert_permute`

b) 解析tiling参数：`GET_TILING_DATA(tilingData, tiling)`从TilingData中获取host侧传入的数据

c) 核心算法实现：

   - 使用SIMT函数`SimtCompute`进行并行计算
   - 每个线程采用网格跨循环(grid-stride loop)的策略处理输入数据

## 单算子测试
算子编译与部署、算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/invert_permute/README.md)。