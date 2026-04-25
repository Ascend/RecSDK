# GR模型NPU适配

## 适配说明

- 本样例的适配对象为 recsys-gr 模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。
- 模型参考的开源链接为 <https://github.com/NVIDIA/recsys-examples/tree/main/examples>。
- 克隆源码并固定版本为: v25.09。

## 运行环境准备

请参考：[模型样例运行环境说明](../../torch_examples/README.md)

## 配套版本

| 配套版本  | PyTorch | torch-npu | torchrec  | dynamic\_emb | fbgemm\_gpu | MindSpeed     | Megatron-LM   |
| ----- | ------- | --------- | --------- | ------------ | ----------- | ------------- | ------------- |
| 配套版本1 | 2.7.1   | 2.7.1     | 1.2.0+npu | 25.09        | 1.2.0       | core\_v0.14.0 | core\_v0.14.0 |

## 安装依赖项

### 1. 模型特定依赖安装

安装gin-config：

```shell
pip3 install gin-config
```

### 2. MindSpeed和Megatron-LM安装

参考 <https://gitcode.com/Ascend/MindSpeed/tree/core_r0.14.0> 下载指定版本的MindSpeed (core\_r0.14.0) 和 Megatron-LM (core\_v0.14.0)，并放置在gr目录下：

```shell
-- gr
   |-- Mindspeed
   |-- Megatron-LM
   |-- recsys-examples
```

### 3. dynamic\_embedding安装

参考[dynamic\_embedding适配说明](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v2/dynamic_emb/README.md)安装dynamic\_embedding。

### 4. 模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。

```shell
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout v25.09
cp -f ../gr_npu.patch ./ && git apply gr_npu.patch
```

## 数据集准备

参考 <https://github.com/NVIDIA/recsys-examples/blob/v25.09/examples/hstu/README.md#dataset-preprocessing> 准备数据集(示例为ml-20m数据集)，并将tmp\_data放置在recsys-examples/examples/hstu路径下：

```shell
-- tmp_data
   |-- ml-20m
```

## 模型运行

### 1. 配置运行环境

拷贝run.sh到hstu目录：

```shell
cp ../run.sh ./examples/hstu/
cd ./examples/hstu
```

### 2. 修改运行参数

修改run.sh的MEGATRON\_DIR和MINDSPEED\_DIR，修改WORLD\_SIZE和ASCEND\_RT\_VISIBLE\_DEVICE为实际使用卡数和卡号。

### 3. 执行训练

环境变量 NPU\_PROFILE 控制是否开启profiling，默认为0。

```shell
bash run.sh
```

## GR模型FSDP2运行说明

### GR模型FSDP2适配说明

FSDP2（Fully Sharded Data Parallel 2）是由PyTorch提供的一种分布式训练策略，它通过将模型参数、梯度和优化器状态在所有参与训练的设备上进行分片，有效减少了内存使用，同时保持了良好的训练性能。本项目支持在NPU上启用FSDP2对recsys-example进行分布式训练。

## 配套版本

| 配套版本  | PyTorch | torch-npu | torchrec  | dynamic\_emb | fbgemm\_gpu | MindSpeed | Megatron-LM   |
| ----- | ------- | --------- | --------- | ------------ | ----------- | --------- | ------------- |
| 配套版本1 | 2.7.1   | 2.7.1     | 1.2.0+npu | 25.09        | 1.2.0       | master    | core\_v0.12.1 |

## 安装依赖项

运行FSDP2模式需要更换MindSpeed及Megatron-LM版本。
参考 <https://gitcode.com/Ascend/MindSpeed/tree/master> 下载主干版本的MindSpeed 和 Megatron-LM (core\_v0.12.1)：

```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
```

## 模型运行

### 1. 配置运行环境

拷贝run\_fsdp2.sh到hstu目录：

```shell
cp ../run_fsdp2.sh ./examples/hstu/
cd ./examples/hstu
```

### 2. 修改运行参数

FSDP2的详细配置位于 `recsys-examples/examples/hstu/configs/fsdp2_config.yaml` 文件中，主要配置项包括：

- `sharding_size`: 分片大小，使用Null表示使用所有可用进程进行分片
- `reshard_after_forward`: 前向传播后是否重新分片参数
- `param_dtype`: 参数数据类型
- `reduce_dtype`: 梯度归约数据类型
- `output_dtype`: 输出数据类型
- `sub_modules_to_wrap`: 需要用FSDP2包装的模块
- `ignored_modules`: 需要忽略的模块

### 3. 执行训练

修改run\_fsdp2.sh的MEGATRON\_DIR和MINDSPEED\_DIR为实际路径，修改WORLD\_SIZE和ASCEND\_RT\_VISIBLE\_DEVICE为实际使用卡数和卡号。

环境变量 NPU\_PROFILE 控制是否开启profiling，默认为0。

```shell
bash run_fsdp2.sh
```
