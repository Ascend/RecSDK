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

## asynchronous_complete_cumsum算子介绍

1. 算子分析

a) 算子的主要功能是实现输入offset的累加；  
b) 算子输入说明：
* x：输入的offset tensor, eg: [1,5,6]

c) 算子输出说明：
* y：输入的offset tensor对应的累加和, eg: [0, 1, 6, 12]

d) 算子约束说明：
* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：int32；int64
* 输入的数据只支持1维。
* 算子参数均会在NPU显存中存放，请根据显存大小合理设置参数长度。


2. Host侧算子实现

Host侧算子实现在目录 op_host下

a) Tiling实现

namespace optiling域中的Tiling函数，主要实现从context中获取外部入参信息（输入参数指针、shape信息），及校验有效性；  
设置BlockDim，最后通过TilingData传递属性信息。

b) Shape推导

推导输出的rShape和DataType函数体。

c) 原型注册

定义了算子原型，并将算子注册到GE。

3. Kernel侧算子实现

Kernel侧算子实现在目录op_kernel下，其中包括：asynchronous_complete_cumsum.cpp。

a) 核函数的入口：`extern "C" __global__ __aicore__ void asynchronous_complete_cumsum`

b) 解析tiling参数：`GET_TILING_DATA(tilingData, tiling)`从TilingData中获取host侧传入的数据

c) 实现累计和的计算

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)
