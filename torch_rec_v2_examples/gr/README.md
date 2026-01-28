# GR模型NPU适配

## 适配说明

本样例的适配对象为 recsys-gr 模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。
模型参考的开源链接为 https://github.com/NVIDIA/recsys-examples/tree/main/examples
克隆源码并固定版本为: v25.09

## 运行环境准备

请参考：[模型样例运行环境说明](../../torch_examples/README.md)

## 安装依赖项：

1. 安装 gin-config
```shell
pip3 install gin-config
```



## 模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。
```shell
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout v25.09
cp -f ../gr_npu.patch ./ && git apply gr_npu.patch
```

## 模型运行

1. 拷贝run.sh到hstu目录：
```shell
cp ../run.sh ./examples/hstu/
cd ./examples/hstu
```
2. 参考 https://gitcode.com/Ascend/MindSpeed/tree/core_r0.14.0 安装 MindSpeed 和 Megatron-LM (core_v0.12.1)
在gr_nv目录下载Mindspeed文件夹和Megatron-LM文件夹
```shell
-- gr_nv
   |-- Mindspeed
   |-- Megatron-LM
   |-- recsys-examples

```

2. 参考 https://github.com/NVIDIA/recsys-examples/blob/v25.09/examples/hstu/README.md#dataset-preprocessing 准备数据集(示例为ml-20m数据集)。

```shell
-- tm_data
   |-- ml-20m
```
tm_data放置在recsys-examples/examples/hstu路径下

3. 执行命令：
修改run.sh的MEGATRON_DIR和MINDSPEED_DIR，修改WORLD_SIZE和ASCEND_RT_VISIBLE_DEVICE为实际使用卡数
和卡号

环境变量 NPU_PROFILE 控制是否开启profiling，默认为0。
```shell
bash run.sh
```