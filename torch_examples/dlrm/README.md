# DLRM模型迁移样例说明

## 适配说明

本样例以DLRM模型为例,适配torchrec框架并在NPU上进行训练。 模型参考的开源链接为:https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/ 克隆源码并固定版本为:Commits on Jun 7 , 2024，提交的SHA-1 hash值（提交ID）：b631a99

## 代码结构说明

```shell
├── dlrm_npu.patch         # 模型迁移适配patch文件
├── generate_data.py       # 随机生成模型数据脚本
├── README.md              # 迁移样例说明文档
└── run.sh                 # 模型运行脚本
```

## 运行环境准备
请参考：RecSDK/torch_examples/README.md


## dlrm源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。
```shell
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99 
cp -f ../dlrm_npu.patch ./
git apply dlrm_npu.patch
```

### 数据集下载
说明：本样例提供两种获取数据集的方式：使用官网数据集可验证模型性能和精度，若仅验证模型功能跑通可使用随机数据集。

1.官网数据集

官网提供两种方式跑通demo：

（1）下载原始数据处理后，提前进行mutil-hot的合成，产生4T的数据

（2）下载原始数据处理后，在训练的过程中生成mutil-hot数据，使用690gb数据集

由于(1)需要的条件苛刻，大部分机器很难满足条件，本次演示使用(2)中的条件。无host瓶颈的情况下，对性能影响较小。需要修改模型脚本代码，让host生成的数据在pin_memory上。

进入[开源模型官网](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD)，按照指引下载数据集到指定目录。该数据集已经托管到HuggingFace:https://huggingface.co/datasets/criteo/CriteoClickLogs 也可直接前往下载。


2.使用生成的数据集
```shell
mkdir generate_data
cp generate_data.py generate_data
cd generate_data
python3 generate_data.py
```

数据集准备完成后的格式如下，后续模型运行时会配置该数据集文件路径。
```shell
day_0_sparse.npy
day_0_dense.npy
day_0_labels.npy
...
day_23_sparse.npy
day_23_dense.npy
day_23_labels.npy
```
说明：数据集较大，数据下载时间较长，请预留时间和磁盘空间，官网数据集大约690GB,随机生成数据集大约71GB


## 修改脚本并运行

修改run.sh文件中的参数，后拷贝到torchrec_dlrm目录下(与dlrm_main.py同级目录)，然后运行模型。

```shell
# 环境参数配置说明（根据实际情况修改）
export PREPROCESSED_DATASET="/path/to/data"                                           # 数据集文件路径
export WORLD_SIZE=8                                                                   # 运行npu卡数，默认8卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                                      # 可用npu卡编号，与WORLD_SIZE数量保持一致

# 运行代码
bash run.sh
```

## 精度、性能对比

为与开源模型比较性能和精度，模型的默认配置参数不建议修改，对比结果如下表所示。

| Device Type | Number of GPUs/NPUs |Collective Size of Embedding Tables (GiB)|Local Batch Size|Global Batch Size|Learning Rate|Interaction Type|Optimizer| AUROC Over Test Set After 1 Epoch | Training speed                        | Time to Train 1 Epoch |Unique Flags|
|-------------|---------------------| --- | --- | --- | --- | --- | --- |-----------------------------------|---------------------------------------|-----------------------| --- |
| GPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7973                            | ~55.0 batches/s == ~901,120 samples/s | 1h20m21s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn` |
| NPU         | 8                   |104.54|2,048|16,384|0.006|DCN v2|Adagrad| 0.7975                            | ~59.0 batches/s == ~966,656 samples/s | 1h12m03s              |`--batch_size 2048 --learning_rate 0.006 --adagrad --interaction_type=dcn`|

说明：NPU测试结果为在参考镜像的X86环境上的测试结果。GPU测试数据参考: https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/ 。
