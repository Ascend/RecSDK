# AttentionFusion融合算子及样例说明

## AttentionFusion融合算子文件结构

```shell
├── aclnn_attention_fusion  # 单算子测试用例
├── attention_fusion.json    # 算子原型配置
├── op_host    # Attention融合算子Host侧实现
├── op_kernel  # Attention融合算子Kernel侧实现
├── README.md  # Attention融合算子说明文档
└── run.sh     # Attention融合算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## AttentionFusion融合算子使用

1. 上传attention_fusion文件夹到目标环境，并进入当前目录，执行指令对attention_fusion融合算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## AttentionFusion融合算子介绍

1. 算子分析

a) 算子的主要功能是实现Attention的正向计算；  
b) 算子输入说明：
* query：query矩阵
* key: key矩阵
* value: value矩阵
* attn_mask: mask矩阵
* mask_on: 可配置mask是否生效，默认为0

c) 算子输出说明：
* atten_score: 算子输出结果
* softmax_out: softmax结果

d) 算子约束说明：
* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC2及之后版本；
* 支持的输入数据类型：float32；
* 输入的数据只支持3维。
* 输入的数据的batch size均相等, 且值在(0, 2000)
* 输入的数据的满足attention的公式，shape支持对应的matmul计算
* 输入的数据除batch size外，所有的维度满足(0, 1000)
* 输入query、key的M必须一致
* 输入value、key的N必须一致
* 输入mask的N, M分别等于query的N，key的M。mask的值只能为0或1。
* mask_on的取值范围为[0, 1], 其中0为不进行mask计算，1为开启mask计算。
* 融合算子的性能提升适用于小算子间free时间较长的情况

2. Host侧算子实现

Host侧算子实现在目录 op_host下

a) Tiling实现

namespace optiling域中的Tiling函数，主要实现从context中获取外部入参信息（输入参数指针、shape信息），及校验有效性；  
并计算kernel侧需要的数据切分相关参数，包括softmax、matmul、ub大小、batch等（详情见tiling文件注释），设置BlockDim，最后通过TilingData传递属性信息。

b) Shape推导

推导输出的rShape和DataType函数体。

c) 原型注册

定义了算子原型，并将算子注册到GE。

3. Kernel侧算子实现

Kernel侧算子实现在目录op_kernel下，其中包括：attention_fusion.cpp。

a) 核函数的入口：extern "C" __global__ __aicore__ void attention_fusion

b) 解析tiling参数：GET_TILING_DATA(tilingData, tiling)从TilingData中获取host侧传入的数据

c) 调用AttentionFusionKernel完成计算；

## AclNN单算子测试参考设计

更多详情可以参考CANN官方的[Ascend C单算子调用概述](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0036.html)。

单算子调用分为两种方式：单算子API执行和模型执行。mxRec提供单算子API执行供参考。

单算子测试用例在目录aclnn_attention_fusion下，其中：

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

### Attention融合算子的AclNN调用实现

调用入口在src/main.cpp中：

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

* gen_data.py：生成AttentionFusion融合算子的输入数据和用于精度校验的golden数据，可自行修改测试相关dim参数。
* verify_result.py：将算子的输出和脚本生成的golden数据进行精度比对，并输出比较结果。比对规则为：允许误差精度loss：1e-4

a) 绝对误差
b) 相对误差
c) 误差相对个数

同时满足绝对误差不全小于loss，相对误差不全小于loss，且绝对误差和相对误差大于loss的个数都超过总数的1/loss，也就是
1/10000（万分之一），即认为算子精度不达标。其余情况均认为算子达标。

用户可自行修改允许精度误差范围loss。