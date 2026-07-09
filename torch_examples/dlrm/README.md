# DLRM模型迁移样例说明

## 适配说明

本样例以DLRM模型为例，适配Rec SDK Torch框架并在NPU上进行训练。

模型参考的[开源模型代码](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/)。 

克隆源码并固定版本为：`Commits on Jun 7, 2024`，提交的SHA-1 Hash值（提交ID）：b631a99。

## 代码结构说明

```shell
├── dlrm_npu.patch         # 模型迁移适配patch文件
├── generate_data.py       # 生成随机数据集脚本
├── README.md              # 样例迁移说明文档
└── run.sh                 # 模型运行脚本
```

## 运行环境准备

请参考[模型样例运行环境说明](../README.md)中“基础镜像”和“启动容器”章节，下载镜像并启动容器。

## dlrm源码适配

克隆当前分支代码，进入**RecSDK/torch_examples/dlrm目录**，下载开源模型代码，并应用patch文件，将开源模型从TorchRec开源框架迁移到基于昇腾NPU的Rec SDK Torch框架。

```shell
# 克隆当前分支
git clone -b develop_examples_and_tools https://gitcode.com/Ascend/RecSDK.git
cd RecSDK/torch_examples/dlrm
# 克隆开源模型源码并做NPU迁移适配
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99
cp -f ../dlrm_npu.patch ./
git apply dlrm_npu.patch && cd -
```

### FAQ

1. 无法访问GitHub网站导致克隆开源模型源码失败

   处理：无法访问GitHub网站时，可考虑从GitCode镜像仓克隆。将上述指令中的：
   
   ```shell
   git clone -b main https://github.com/facebookresearch/dlrm.git
   ```
   
   替换为如下指令并重新执行：

   ```shell
   git clone -b main https://gitcode.com/gh_mirrors/dl/dlrm.git
   ```

2. 容器内执行git指令或Python脚本时报错
   
   参考[容器内执行git指令或Python脚本时报错](../README.md#container_git_python_error)处理。

## 数据集准备

>[!NOTE]
>
> 1. 本样例提供两种获取数据集方式：官网数据集（Criteo Click Logs数据集）和随机生成数据集。使用官网数据集可验证模型性能和精度，使用随机生成数据集可跑通模型基础功能。
> 2. 由于官网数据集（Criteo Click Logs数据集）在[HuggingFace](https://huggingface.co/datasets/criteo/CriteoClickLogs)上归档的文件形式在2026年5月存在更新，但[开源模型社区](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD#running-the-mlperf-dlrm-v2-benchmark)资料中的数据集下载和预处理脚本暂未同步更新，**因此当前建议使用随机生成数据集运行模型**。

- 随机生成数据集

    进入generate_data.py文件目录（RecSDK/torch_examples/dlrm），并运行如下指令，生成约71GB的随机数据集：

    ```shell
    mkdir generate_data
    cp generate_data.py generate_data
    cd generate_data && python3 generate_data.py && cd -
    ```

    生成随机数据集耗时约3~5min，可能随可用CPU资源而变化。

- 官网数据集（Criteo Click Logs数据集）

    开源模型社区提供两种基于官网数据集跑通DLRM模型的方式：

    方式1：下载原始数据处理后，提前进行multi-hot的合成，生成4TB大小的数据集。

    方式2：下载原始数据处理后，在训练的过程中生成multi-hot数据，共690GB大小的数据集。

    > 由于“方式1”需要的条件苛刻，大部分机器很难满足，后续启动脚本将基于“方式2”运行模型。“方式2”在无host瓶颈的情况下，对性能影响较小。

    官网数据集下载和预处理：

    参考[开源模型社区 - 运行DLRM v2模型](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD#running-the-mlperf-dlrm-v2-benchmark)，按照“Step 1”和“Step 2”章节下载数据集并进行预处理。其中“Step 2”的输出结果即为“方式2”需要的数据集。

    该数据集已经托管到[HuggingFace](https://huggingface.co/datasets/criteo/CriteoClickLogs)，也可直接前往下载。

    官网数据集较大，数据下载时间较长，请提前预留磁盘空间。

数据集准备完成后，**需记录数据集文件路径**，后续模型运行前会配置该数据集文件路径。

数据集格式如下：

```shell
day_0_sparse.npy
day_0_dense.npy
day_0_labels.npy
...
day_23_sparse.npy
day_23_dense.npy
day_23_labels.npy
```

## 修改脚本并运行模型

1.进入**RecSDK/torch_examples/dlrm**目录，并修改目录下run.sh文件中的如下参数：

```shell
# 环境参数配置说明（根据实际情况修改）
export PREPROCESSED_DATASET="../../generate_data"  # 数据集文件路径。必选，需修改为实际数据集文件路径。
export WORLD_SIZE=2                             # 运行npu卡数，默认2卡。可选。
export ASCEND_RT_VISIBLE_DEVICES=0,1            # 可用npu卡编号，与WORLD_SIZE数量保持一致。可选。
```

说明：本样例默认使用随机生成数据集在2卡(0,1)环境运行。若使用官网数据集训练，设备需满足8 NPU卡 * 64GB环境，并修改run.sh中WORLD_SIZE和ASCEND_RT_VISIBLE_DEVICES环境变量：`export WORLD_SIZE=8`，`export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

2.拷贝run.sh文件到开源模型代码dlrm/torchrec_dlrm目录下

```bash
cp run.sh dlrm/torchrec_dlrm
```

进入开源模型代码dlrm/torchrec_dlrm目录并运行模型

```bash
cd dlrm/torchrec_dlrm
bash run.sh
```

预期耗时：使用随机生成数据集，默认配置下，耗时约5~10min。

预期输出：训练结束后出现`Job finished: SUCCEEDED`字样，说明模型训练完成。若使用随机生成数据集进行训练，训练结束时打印的`Test AUROC`值仅做参考。

## 性能数据采集并分析

1.修改run.sh文件中如下参数

```shell
# 环境参数配置说明（根据实际情况修改）
export ENABLE_PROF=1 # 开启性能采集模式
export PROF_START=50 # 开始采集步数，不建议从最开始就采集数据，模型训练稳定后开始采集数据更准确。
export PROF_STEP=10  # 采集步数，训练时每个step运行功能重复，没必要采集过多步数。
```

修改run.sh脚本后重新执行训练脚本即可采集到性能数据用于分析，性能数据默认保存在运行目录./profiler目录。具体性能分析方法请参考[昇腾社区-训练推理开发工具-模型调优msProf章节](https://www.hiascend.com/document/detail/zh/mindstudio/2600/msTT_msIT/ascend_pytorch_profiler/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)。

## 多级缓存模式

修改run.sh中MODES参数为"embcache"即可切换模型为"稀疏表多级缓存"模式，然后重新执行训练脚本。

```shell
MODES="embcache"
```

## 精度、性能对比

说明：为与开源模型比较性能和精度，模型需基于8卡环境运行。其中NPU测试结果为x86环境上且使用官网数据集并配置run.sh中的`MODES="hybrid_torchrec"`模式的测试结果。GPU测试数据参考[开源模型社区](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/) 。 对比结果如下表所示：

| Device Type | Number of GPUs/NPUs |Collective Size of Embedding Tables (GiB)|Local Batch Size|Global Batch Size|Learning Rate|Interaction Type|Optimizer| AUROC Over Test Set After 1 Epoch | Training speed                        | Time to Train 1 Epoch |Unique Flags|
|-------------|---------------------| --- | --- | --- | --- | --- | --- |-----------------------------------|---------------------------------------|-----------------------| --- |
| GPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7973                            | ~55.0 batches/s == ~901,120 samples/s | 1h20m21s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn` |
| NPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7975                            | ~59.0 batches/s == ~966,656 samples/s | 1h12m03s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn`|
