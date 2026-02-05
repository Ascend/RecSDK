**说明**

本算子仅支持NPU调用

# 产品支持情况
| 硬件型号              | 是否支持 |
| -------------------- |------|
| Atlas A2训练系列产品  | 是    |
| Atlas A3训练系列产品  | 是    |
| Atlas 推理系列产品    | 是    |


# AsynchronousCompleteCumsum算子目录层级

```shell
-- asynchronous_complete_cumsum
   |-- v220
      |-- op_host                              # 算子host侧实现
      |-- op_kernel                            # 算子kernel侧实现
      |-- asynchronous_complete_cumsum.json    # 算子原型配置
      |-- README.md                            # 算子说明文档
      |-- run.sh                               # 算子编译部署脚本
```

# 功能
对输入的一维Tensor累积求和

# 算子实现原理
输入:
```python
input_tensor = [1,5,6]
```
输出：
```python
output_tensor = [0, 1, 6, 12]
```

1. Host侧算子实现

Host侧算子实现在目录 op_host下

a) Tiling实现

namespace optiling域中的Tiling函数，主要实现从context中获取外部入参信息（输入参数指针、shape信息），及校验有效性；  
设置BlockDim，最后通过TilingData传递属性信息。

b) Shape推导

推导输出的rShape和DataType函数体。

c) 原型注册

定义了算子原型，并将算子注册到GE。

2. Kernel侧算子实现

Kernel侧算子实现在目录op_kernel下，其中包括：asynchronous_complete_cumsum.cpp。

a) 核函数的入口：`extern "C" __global__ __aicore__ void asynchronous_complete_cumsum`

b) 解析tiling参数：`GET_TILING_DATA(tilingData, tiling)`从TilingData中获取host侧传入的数据

c) 实现累计和的计算

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/asynchronous_complete_cumsum/README.md)