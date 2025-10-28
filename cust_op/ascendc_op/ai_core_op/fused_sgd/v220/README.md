# SGD优化器融合算子及样例说明

## SGD融合算子文件结构

```shell
├── sgd.json    # 算子原型配置
├── op_host    # SGD融合算子Host侧实现
├── op_kernel  # SGD融合算子Kernel侧实现
├── README.md  # SGD融合算子说明文档
└── run.sh     # SGD融合算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## SGD融合算子使用

1. 上传fused_sgd文件夹到目标环境，并进入当前目录，执行指令对sgd融合算子进行编译和部署。默认编译安装Atlas A2训练系列产品AI Core类型。

```shell
bash run.sh
```

若指定 AI Core 类型编译：

```shell
bash run.sh ai_core-<soc_version>
```
> AI处理器的型号<soc_version>请通过如下方式获取:
> - 在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，获取`Chip Name`信息。实际配置值为AscendChip Name，例如`Chip Name`取值为`xxxyy`，实际配置值为`Ascendxxxyy`。
>
> 基于同系列的AI处理器型号创建的算子工程，其基础功能（基于该工程进行算子开发、编译和部署）通用。

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 模型脚本中创建sgd优化器并指定使用融合算子实现。代码示例：

```python
from mx_rec.optimizers.gradient_descent import create_hash_optimizer

# 创建sgd优化器时增加"use_fusion_optim=True"参数，表示使用融合算子实现。use_fusion_optim参数默认值为False。
# sgd优化器详细使用指导请参考Rec SDK用户指南。
sparse_optimizer = create_hash_optimizer(learning_rate=0.001, use_fusion_optim=True)
```

## SGD融合算子介绍
1. 算子分析  
a) 算子的主要功能是实现sgd优化器反向更新时参数的计算和更新；  
b) 算子参数说明：

* gradient: sgd优化器计算时使用的梯度；
* indices: 参与计算/更新的数据索引；
* inputVar: embedding表对应的variable数据；计算结果原地更新；
* lr: 学习率；

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC1及之后版本；
* 支持的输入数据类型：float32；
* embedding表的dim值需要是8的倍数；

2. Host侧算子实现
Host侧算子实现在目录 fused_sgd/v220/op_host下，其中包括：sgd.cpp和sgd_tiling.h。

a) Tiling实现  
在 sgd_tiling.h 文件中，定义了 SgdTilingData 结构体，用于存储算子的切分参数。在 sgd.cpp 文件中，namespace optiling 域中的 TilingFunc 函数主要实现从 context 中获取外部入参信息（输入参数指针、shape信息），并校验其有效性。
此外，该函数还计算了kernel侧需要的数据切分相关参数，包括 batchSize、tableSize、dimSize、actualCoreNum、ubFreeSize、splitNextCoreProcBs、splitPrevCoreProcBs 和 splitCoreIndex。最后，通过 TilingData 传递这些属性信息。

b) Shape推导

namespace ge 域中的 InferShape 和 InferDataType 函数用于推导算子的输出形状和数据类型。算子计算结果原地更新到输入参数中。

c) 原型注册

namespace ops 域中的 Sgd 类定义了算子原型，并将算子注册到GE。

3. Kernel侧算子实现

Kernel侧算子实现在目录fused_sgd/v220/op_kernel下，其中包括：sgd.cpp、sgd_kernel.h和sgd_kernel_base.h。

a) 核函数的入口：extern "C" __global__ __aicore__ void sgd

b) 解析tiling参数：GET_TILING_DATA(tiling_data, tiling)：从TilingData中获取host侧传入的数据。

c) Init方法，用于初始化算子运行所需的数据；

d) Process方法，进行数据搬入和计算，并将计算结果更新到对应的输入参数中。

## 单算子测试
详见[README.md](../../../../test/sgd_test/tf/README.MD)。