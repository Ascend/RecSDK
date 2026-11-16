# GR模型NPU适配

## 适配说明

本样例的适配对象为 recsys-gr 模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。

模型参考的[开源代码](https://github.com/NVIDIA/recsys-examples/tree/main/examples)。

克隆源码并固定版本为: v25.05。

验证运行的算力平台：Atlas A2训练系列产品。

## 运行环境准备

请参考：[模型样例运行环境说明](../README.md)

## 安装依赖项

安装gin-config和torchvision。torchvision需安装和PyTorch配套版本，PyTorch 2.7.1对应的torchvision版本为0.22.1。

```shell
pip3 install gin-config
pip3 install torchvision==0.22.1
```

### 安装NPU算子包

本样例依赖`fbgemm_ascend`和`ops_rec`两个NPU算子包。

- `fbgemm_ascend`：安装方法请参考[安装自定义算子相关包](../README.md#3-安装自定义算子相关包)中的 3.1 节。

- `ops_rec`：从[ops-rec](https://gitcode.com/Ascend/ops-rec)源码编译安装。算子依赖 CATLASS 源码，编译前需初始化 submodule：

  ```shell
  git submodule update --init --recursive
  bash build.sh --socs atlas_a2 --install -j16
  ```

  默认编译`atlas_950`、`atlas_a2`、`atlas_a3`三个SOC变体；也可只编译当前平台，例如 A2：

  ```shell
  bash build.sh --socs atlas_a2 --install -j16
  ```

  生成的 whl 位于`dist/`，也可以手动安装：

  ```shell
  pip3 uninstall -y ops_rec
  pip3 install dist/ops_rec-*.whl
  ```

安装后`import ops_rec`会通过`acl.get_soc_name()`自动检测芯片型号并加载对应 SOC 变体的算子，无需手工设置`ASCEND_CUSTOM_OPP_PATH`。

本例运行在 Atlas A2 上，使用 `ops_rec` 的 HSTU v1 算子。前向接口为 `torch.ops.ops_rec.hstu_jagged.equal`，反向由 `hstu_jagged_backward.equal` 提供，`import ops_rec` 时会自动注册自动微分，训练中直接调用前向并 `loss.backward()` 即可。

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

```
git clone -b core_r0.8.0 https://gitcode.com/Ascend/MindSpeed.git
git clone -b core_r0.8.0 https://github.com/NVIDIA/Megatron-LM.git
```

安装后gr_nv目录下文件结构如下：

```shell
-- gr_nv
   |-- Mindspeed
   |-- Megatron-LM
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

run.sh中的WORLD_SIZE和ASCEND_RT_VISIBLE_DEVICE为实际使用卡数和卡号，可根据实际情况修改。

执行如下命令运行模型：

```shell
bash run.sh
```

### 精度验证

默认关闭。设置 `PRECISION_FLAG=1` 可开启精度验证模式：固定随机种子（msprobe `seed_all`）、关闭数据集 shuffle，并将模型参数初始化为固定值，保证同一数据集多次训练结果一致。该模式依赖 `msprobe`（MindStudio 精度工具）。

开启后训练结束会将前 100 步的 loss 落盘到 `../save_results_NPU/precision_losses.csv`（相对 examples/hstu 目录），两次训练分别保存该文件后进行对比即可：

```shell
PRECISION_FLAG=1 bash run.sh
```
