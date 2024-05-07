# LazyAdam优化器融合算子及样例说明

## LazyAdam融合算子文件结构

```shell
├── aclnn_lazy_adam_test  # 单算子测试用例
├── lazy_adam.json    # 算子原型配置
├── op_host    # LazyAdam融合算子Host侧实现
├── op_kernel  # LazyAdam融合算子Kernel测实现
├── README.md  # LazyAdam融合算子说明文档
└── run.sh     # LazyAdam融合算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## lazy_adam融合算子使用

1. 上传fused_lazy_adam文件夹到目标环境，并进入当前目录，执行指令对lazy_adam融合算子进行编译和部署

```shell
bash run.sh
```

2. 模型脚本中创建lazy_adam优化器并指定使用融合算子实现。代码示例：

```python
from mx_rec.optimizers.lazy_adam import create_hash_optimizer

# 创建lazy_adam优化器时增加"use_fusion_optim=True"参数，表示使用融合算子实现。use_fusion_optim参数默认值为False。
# lazy_adam优化器详细使用指导请参考mxRec用户指南。
sparse_optimizer = create_hash_optimizer(learning_rate=0.001, use_fusion_optim=True)
```

## LazyAdam融合算子

1. 算子分析

a) 算子的主要功能是实现lazy_adam优化器反向更新时m、v、variable三项数据的计算和更新；  
b) 算子参数说明：

* gradient: lazy_adam优化器计算时使用的梯度；
* indices: 参与计算/更新的数据索引；
* inputM: lazy_adam优化器一阶矩估计；计算结果原地更新；
* inputV: lazy_adam优化器二阶矩估计；计算结果原地更新；
* inputVar: embedding表对应的variable数据；计算结果原地更新；
  c) 算子约束说明：
* 支持的型号：Atlas A2系列产品;
* 支持的输入数据类型：float32；
* embedding表的dim值需要是8的倍数；

2. Host侧算子实现

Host侧算子实现在目录 fused_lazy_adam/op_host下，其中包括：lazy_adam.cpp和
lazy_adam_tiling.h。

a) Tiling实现

namespace
optiling域中的LazyAdamTilingFunc函数，主要实现从context中获取外部入参信息（输入参数指针、shape信息），及校验有效性；  
并计算kernel侧需要的数据切分相关参数，包括row、loopCount、batch等（详情见tiling文件注释），设置BlockDim，最后通过TilingData传递属性信息。

b) Shape推导

因算子计算结果原地更新到输入参数中，namespace ge域中的InferShape和InferDataType函数体为空。

c) 原型注册

namespace ops域中的LazyAdam类定义了算子原型，并将算子注册到GE。

3. Kernel侧算子实现

Kernel侧算子实现在目录fused_lazy_adam/op_kernel下，其中包括：lazy_adam.cpp。

a) 核函数的入口：extern "C" __global__ __aicore__ void lazy_adam

b) 解析tiling参数：GET_TILING_DATA(tilingData, tiling)从TilingData中获取host侧传入的数据

c) Init方法，进行算子运行数据的初始化；

d) Process方法，进行数据搬入和计算，并且计算完成后将计算结果数据分别更新到对应入参中；

## AclNN单算子测试参考设计

更多详情可以参考CANN官方的[Ascend C单算子调用概述](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0036.html)。

单算子调用分为两种方式：单算子API执行和模型执行。mxRec提供单算子API执行供参考。

单算子测试用例在目录fused_lazy_adam/aclnn_lazy_adam_test下，其中：

* inc是头文件目录
* scripts存放生成数据和验证数据的python脚本
* input是存放算子入参的bin文件
* output是存放生成的可执行程序execute_op、算子输出bin文件和用于验证的golden数据bin文件
* src是存放公共函数common、构造算子输入输出描述类oprator_desc、单算子调用主体流程实现op_runner文件和入口main文件

执行单算子测试：

```shell
bash run.sh
```

### 前置条件

1.

参考[基于msopgen工具创建算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0023.html)
完成算子工程的创建，
参考[kernel侧算子实现](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0024.html)
完成kernel侧实现的相关准备，
参考[host侧算子实现](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0026.html)
完成host侧实现相关准备。

2.

参考[算子编译部署](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0031.html)
完成算子的编译部署，编译部署时需要开启算子的二进制编译功能：修改算子工程中的编译配置项文件CMakePresets.json，将
ENABLE_BINARY_PACKAGE设置为True。编译部署时可将算子的二进制部署到当前环境，便于后续算子的调用。

3.

检查API执行需要的头文件和库文件是否自动生成，针对融合算子，检查cust_op/fused_lazy_adam/lazy_adam/build_out/autogen目录下，是否有
aclnn_lazy_adam.cpp和aclnn_lazy_adam.h等。

注意：对于cust_op/fused_lazy_adam/run.sh脚本，安装算子后会删除构建目录。运行单算子测试时，需要屏蔽掉删除rm rf
./lazy_adam这一步，以确保前置条件3。

### LazyAdam融合算子de AclNN调用实现

针对LazyAdam融合算子，入口src/main.cpp中：

1. InitResource函数：初始化AscendCL并运行管理资源申请，不用修改
2. RunLookupOp运行算子：

a) 创建算子输入输出描述CreateOpDesc，OperatorDesc对象定义(inc/operator_desc.h)中设置了算子入参为成员变量，以便后续
op_runner中使用；

b) 创建OpRunner的对象，并依次执行：

* opRunner.Init()：申请内存存放执行算子的输入输出数据
* SetInputData()：加载数据输入bin文件并传输给OpRunner的Buffer供后续算子执行使用
* opRunner.RunOp()：算子执行，主要流程为：入参数据拷贝，创建Stream，执行Stream，输出数据拷贝，释放Stream资源
* ProcessOutputData()：算子输出数据处理，并落盘文件，以供后续与golden数据比对

3. DestroyResource函数：释放内存，不用修改

### 运行脚本

run.sh脚本依次执行：

1. 清除遗留生成文件和日志文件
2. 生成输入数据和真值数据
3. 编译acl可执行文件
4. 运行可执行文件
5. 比较真值文件

### scripts脚本

* gen_data.py：生成LazyAdam融合算子的输入数据和用于精度校验的golden数据，可自行修改测试相关dim参数。
* verify_result.py：将算子的输出和脚本生成的golden数据进行精度比对，并输出比较结果。比对规则为：允许误差精度loss：1e-6

a) 绝对误差
b) 相对误差
c) 误差相对个数

同时满足绝对误差不全小于loss，相对误差不全小于loss，且绝对误差和相对误差大于loss的个数都超过总数的1/loss，也就是
1/1000000（百万分之一），即认为算子精度不达标。其余情况均认为算子达标。

用户可自行修改允许精度误差范围loss。