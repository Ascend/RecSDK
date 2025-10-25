# 稀疏表自动扩容算子及样例说明

## 扩容算子文件结构
```shell
├── emb_custom.json    # 算子配置
├── op_host    # 扩容算子Host侧实现
├── op_kernel  # 扩容算子Kernel测实现
├── README.md  # 扩容算子说明文档
└── run.sh     # 扩容算子安装脚本
```

## Ascend C参考设计
更多详情可以参考CANN官方的Ascend C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html)。

针对Rec SDK，用于动态扩容功能的Ascend C算子有两个：**查询算子embedding_lookup_by_addr**和**更新算子embedding_update_by_addr**，
以下以embedding_lookup_by_addr算子为例对扩容算子做详细说明，embedding_update_by_addr算子同理。

## 查询算子embedding_lookup_by_addr

1. 算子分析

a) 算子的主要功能是用addr地址作为入参，替换tf.gather算子；

b) 算子支持emb表为int32、float32和float16三种类型的emb查询；

c) 算子入参为：表示待查询emb地址列表的address，表示待查询emb的维度embedding_dim，表示待查询emb的类型embedding_type，
其中，0：int32、1：float32、2：float16.

2. Host侧算子实现

Host侧算子实现在目录cust_op_by_addr/op_host下，其中包括：embedding_lookup_by_address.cpp和
embedding_lookup_by_address_tiling.h。

a) Tiling实现 

namespace optiling域中的TilingFunc函数，主要实现从context中获取外部入参信息，并校验有效性，并计算kernel侧需要的中间变量，如embeddingDimAligned、
addrPerLoop等，设置BlockDim，最后通过TilingData传递属性信息。

b) Shape推导

namespace ge域中的InferShape和InferDataType函数，主要通过输入的tensorShape和tensorType来推导输出的tensorShape和tensorType。

c) 原型注册

namespace ops域中的EmbeddingLookupByAddress类。

3. Kernel侧算子实现

Kernel侧算子实现在目录cust_op_by_addr/op_kernel下，其中包括：embedding_lookup_by_address.cpp。

a) 核函数的入口 extern "C" __global__ __aicore__ void embedding_lookup_by_address

b) GET_TILING_DATA(constData, tiling)从TilingData中获取host侧传入的数据

c) 根据模板类KernelEimtable构建类型不同的op对象，依次调用Init_param、Init、Process三个函数实现数据的搬运和计算；

d) KernelEimtable::Init_param函数中，使用获取到的TilingData计算得到singleCoreAddrNum、veclen等变量

e) KernelEimtable::Init函数中，针对非对齐shape算子，使用Init_param的中间变量计算得到每个核上的偏移量、每个分块大小，并初始化和绑定Buffer

f) KernelEimtable::Process函数实现算子的搬运和计算，最终输出结果到dstDataGm，即GM_ADDR y

## AclNN单算子测试参考设计

更多详情可以参考CANN官方的[Ascend C单算子调用概述](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0036.html)。

单算子调用分为两种方式：单算子API执行和模型执行。Rec SDK提供单算子API执行供参考。

单算子测试用例在目录cust_op/test/aclnn_cust_op_by_addr_test下，其中：
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

1. 参考[基于msopgen工具创建算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0023.html)完成算子工程的创建，
参考[kernel侧算子实现](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0024.html)完成kernel侧实现的相关准备，
参考[host侧算子实现](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0026.html)完成host侧实现相关准备。
2. 参考[算子编译部署](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0031.html)完成算子的编译部署，编译部署时需要开启算子的二进制编译功能：修改算子工程中的编译配置项文件CMakePresets.json，将
ENABLE_BINARY_PACKAGE设置为True。编译部署时可将算子的二进制部署到当前环境，便于后续算子的调用。
3. 检查API执行需要的头文件和库文件是否自动生成，针对Rec SDK，检查cust_op/ascendc_op/ai_core_op/cust_op_by_addr/v220/cust_op_by_addr/build_out/autogen目录下，是否有
aclnn_embedding_lookup_by_address.cpp和aclnn_embedding_lookup_by_address.h等。

### 查询算子 embedding_lookup_by_addr
针对embedding_lookup_by_addr算子，入口src/main.cpp中：

1. InitResource函数：初始化AscendCL并运行管理资源申请，不用修改
2. RunLookupOp运行算子：

a) 创建算子输入输出描述CreateOpDescLookup，该类是继承OperatorDesc，主要是引入了embeddingDim和embeddingType两个入参成员变量，以便后续
op_runner中使用，基类OperatorDesc不用做修改；

b) 创建OpRunnerLookup的对象，并依次执行：
* opRunner.Init()：申请内存存放执行算子的输入输出数据
* SetLookupInputData()：加载数据输入bin文件并传输给OpRunner的Buffer供后续算子执行使用
* RunOp()：算子执行，核心调用OpRunnerLookup::RunOpHelper
* ProcessLookupOutputData()：算子输出数据处理，并落盘文件，以供后续与golden数据比对

OpRunnerLookup类重载了基类OpRunner的虚函数RunOpHelper，实现具体算子的aclnn调用，基类OpRunner不用做修改；

3. DestoryResource函数：释放内存，不用修改

### 运行脚本
run.sh脚本依次执行：
1. 清除遗留生成文件和日志文件 
2. 生成输入数据和真值数据 
3. 编译acl可执行文件 
4. 运行可执行文件 
5. 比较真值文件

### scripts脚本
* gen_data.py：生成embedding_lookup_by_addr算子（这里以embedding_lookup_by_addr算子为例）的输入数据和用于精度校
验的golden数据，用户可自行修改测试的规模，如表的大小、查询的数量、表的dim等信息。
* verify_result.py：将算子的输出和脚本生成的golden数据进行精度比对，比对规则为：允许误差精度loss：1e-4

a) 绝对误差
b) 相对误差
c) 误差相对个数

同时满足绝对误差不全小于loss，相对误差不全小于loss，且绝对误差和相对误差大于loss的个数都超过总数的1/loss，也就是
1/10000（双万分之一），即认为算子精度不达标。其余情况均认为算子达标。

用户可自行修改允许精度误差范围loss。