# GR模型NPU适配

## 适配说明

本样例的适配对象为 recsys-gr 模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。

模型参考的[开源代码](https://github.com/NVIDIA/recsys-examples/tree/main/examples)。

克隆源码并固定版本为: v25.05。

验证运行的算力平台：Atlas A2训练系列产品。

## 运行环境准备

请参考：[模型样例运行环境说明](../README.md)

## 安装依赖项

安装 gin-config

```shell
pip3 install gin-config
```

## 模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。

```shell
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout v25.05
cp -f ../gr_nv2npu.patch ./ && git apply gr_nv2npu.patch
```

## 模型运行

### 拷贝run.sh到hstu目录
   
```shell
cp ../run.sh ./examples/hstu/
cd ./examples/hstu
```

### 安装 MindSpeed 和 Megatron-LM

请参考[链接](https://gitcode.com/Ascend/MindSpeed/tree/core_r0.8.0)进行安装。

在gr_nv目录下载Mindspeed文件夹和Megatron-LM文件夹

```shell
-- gr_nv
   |-- Mindspeed
   |-- Megatron-LM
   |-- recsys-examples

```

### 准备数据集

请参考[链接](https://github.com/NVIDIA/recsys-examples/blob/v25.05/examples/hstu/README.md#dataset-preprocessing)准备数据集，示例为ml-20m数据集。

```shell
-- tm_data
   |-- KuaiRand-Pure
      |-- LICENSE                               # LICENSE
      |-- data                                  # 数据
      |-- load_data_pure.py                     # 加载文件

```

tm_data放置在recsys-examples/examples/hstu路径下。

### 执行命令

修改run.sh的MEGATRON_DIR和MINDSPEED_DIR，修改WORLD_SIZE和ASCEND_RT_VISIBLE_DEVICE为实际使用卡数和卡号。

修改后，执行如下命令运行模型：

```shell
bash run.sh
```
