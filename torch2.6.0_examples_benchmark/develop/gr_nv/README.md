# GR模型NPU适配

## 适配说明

本样例的适配对象为 recsys-gr 模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。
模型参考的开源链接为 https://github.com/NVIDIA/recsys-examples/tree/main/examples
克隆源码并固定版本为: v25.05
验证运行的算力平台：Atlas A2训练系列产品

## 运行环境准备

请参考：https://gitcode.com/Ascend/RecSDK/tree/develop/torch_examples/README.md

## 安装依赖项：

1. 安装 gin-config
```shell
pip3 install gin-config
```

2. 参考 https://gitcode.com/Ascend/MindSpeed/tree/core_r0.8.0 安装 MindSpeed 和 Megatron-LM

## 模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。
```shell
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout v25.05
cp -f ../gr_nv2npu.patch ./ && git apply gr_nv2npu.patch
```

## 模型运行

1. 拷贝run.sh到hstu目录：
```shell
cp ../run.sh ./examples/hstu/
cd ./examples/hstu
```

2. 参考 https://github.com/NVIDIA/recsys-examples/blob/v25.05/examples/hstu/README.md#dataset-preprocessing 准备数据集(示例为ml-20m数据集)。

3. 执行命令：
```shell
bash run.sh
```
