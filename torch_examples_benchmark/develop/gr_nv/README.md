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

### 安装NPU算子包

本样例依赖`fbgemm_ascend`和`rec_cust_ops`两个NPU算子包，安装方法请参考[安装自定义算子相关包](../README.md#3-安装自定义算子相关包)。

## 模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。

```shell
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout v25.05
cp -f ../gr_nv2npu.patch ./ && git apply gr_nv2npu.patch && cd -
```

## 模型运行

### 安装MindSpeed和Megatron-LM

请在README同层级目录，参考[链接](https://gitcode.com/Ascend/MindSpeed/tree/core_r0.8.0)进行MindSpeed 和 Megatron-LM的安装。

安装后gr_nv目录下文件结构如下：

```shell
-- gr_nv
   |-- Mindspeed
   |-- Mindspeed
   |-- Megatron-LM
   |-- Megatron-LM
   |-- recsys-examples
   |-- recsys-examples
   |-- gr_nv2npu.patch
   |-- README.md
   |-- run.sh
```

### 拷贝run.sh到hstu目录并进入hstu目录

进入README同层级目录，再执行如下指令：

```shell
cp run.sh ./recsys-examples/examples/hstu/
cd recsys-examples/examples/hstu/
```

### 准备数据集

请参考[链接](https://github.com/NVIDIA/recsys-examples/blob/v25.05/examples/hstu/README.md#dataset-preprocessing)准备数据集，数据集名称选择kuairand-pure。如下示例为kuairand-pure数据集处理后的文件结构。

```shell
-- tmp_data
   |-- KuaiRand-Pure
      |-- LICENSE                               # LICENSE
      |-- data                                  # 数据
      |-- load_data_pure.py                     # 加载文件
```

注：tmp_data放置在recsys-examples/examples/hstu路径下。

### 执行命令

若前面步骤中安装的MindSpeed和Megatron-LM不在gr_nv/目录下，则需将run.sh中的MINDSPEED_DIR/MEGATRON_DIR的值替换为实际MindSpeed/Megatron-LM所在路径。

修改后，执行如下命令运行模型：
run.sh中的WORLD_SIZE和ASCEND_RT_VISIBLE_DEVICE为实际使用卡数和卡号，可根据实际情况修改。

执行如下命令运行模型：

```shell
bash run.sh
```
