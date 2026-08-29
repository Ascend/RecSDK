# 代码结构

```shell
|-- benchmark/
    |-- ckpt/        # 权重文件目录
    |-- configs/     # 框架相关配置文件目录
    |-- data/        # 数据集目录
    |-- models/      # 存放模型代码、模型规格配置文件的目录
    |-- patches/     # 模型迁移适配patch文件目录
    |-- test/        # 模型测试脚本目录
    |-- tools/       # 工具类脚本目录
    |-- README.md    # 模型迁移说明文档
    |-- run.py       # 模型运行脚本
```

# 运行环境准备

参考[RecSDK-Torch 模型样例运行环境说明](../torch_examples_benchmark/develop/README.md)

# 必要依赖安装

```shell

apt-get install protobuf-compiler protobuf-devel
pip install pytest
```

由于DeepCTR仓中tf和torch混用，所以DeepCTR仓模型需要额外安装tf。

```shell
pip install tensorflow
```

# quick start

xxx.json替換为configs目录下的配置文件名；

```shell
python run.py xxx.json --eager
```

**xxx.json**:为configs文件夹中的配置文件\
**--eager**：为强制跑eager模式，不配置默认跑inductor模式（当前未支持）\
**--no\_hf32**：禁用混合精度加速，不配置默认使能混合精度加速。NCF、DIN-pytorch、Multitask-Recommendation-Library、SASRec比对GPU与NPU精度时需禁用\
**--custom\_dropout**：使用自定义的dropout函数，不配置使用默认的dropout函数。Multitask-Recommendation-Library比对GPU与NPU精度时需配置

# Model List

|Model Name|配置文件|
|--|--|
|AFM|[AFM.json](configs/AFM.json)|
|AFN|[AFN.json](configs/AFN.json)|
|AutoInt|[AutoInt.json](configs/AutoInt.json)|
|BERT4Rec|[BERT4Rec.json](configs/BERT4Rec.json)|
|DAT|[DAT.json](configs/DAT.json)|
|DBMTL|[DBMTL.json](configs/DBMTL.json)|
|DCN|[DCN.json](configs/DCN.json)|
|DCNv2|[DCNv2.json](configs/DCNv2.json)|
|DeepFM|[DeepFM.json](configs/DeepFM.json)|
|DIEN|[DIEN.json](configs/DIEN.json)|
|DIFM|[DIFM.json](configs/DIFM.json)|
|DIN|[DIN.json](configs/DIN.json)|
|DIN-pytorch|[DIN-pytorch.json](configs/DIN-pytorch.json)|
|DLRM|[DLRM.json](configs/DLRM.json)|
|DLRM_META|[DLRM_META.json](configs/DLRM_META.json)|
|DMR|[DMR.json](configs/DMR.json)|
|DSSM|[DSSM.json](configs/DSSM.json)|
|EDCN|[EDCN.json](configs/EDCN.json)|
|ESMM|[ESMM.json](configs/ESMM.json)|
|EulerNet|[EulerNet.json](configs/EulerNet.json)|
|FiBiNET|[FiBiNET.json](configs/FiBiNET.json)|
|GRU4Rec|[GRU4Rec.json](configs/GRU4Rec.json)|
|HSTU_META_1M|[HSTU_META_1M.json](configs/HSTU_META_1M.json)|
|HSTU_META_1M_LARGE|[HSTU_META_1M_LARGE.json](configs/HSTU_META_1M_LARGE.json)|
|HSTU_META_20M|[HSTU_META_20M.json](configs/HSTU_META_20M.json)|
|HSTU_META_20M_LARGE|[HSTU_META_20M_LARGE.json](configs/HSTU_META_20M_LARGE.json)|
|IFM|[IFM.json](configs/IFM.json)|
|MIND|[MIND.json](configs/MIND.json)|
|MMOE|[MMOE.json](configs/MMOE.json)|
|MaskNet|[MaskNet.json](configs/MaskNet.json)|
|MultiTower|[MultiTower.json](configs/MultiTower.json)|
|Multitask-Recommendation-Library|[Multitask-Recommendation-Library.json](configs/Multitask-Recommendation-Library.json)|
|NCF|[NCF.json](configs/NCF.json)|
|NFM|[NFM.json](configs/NFM.json)|
|ONN|[ONN.json](configs/ONN.json)|
|PLE|[PLE.json](configs/PLE.json)|
|PNN|[PNN.json](configs/PNN.json)|
|RANKMIXER|[RANKMIXER.json](configs/RANKMIXER.json)|
|RECSYS_RANKING|[RECSYS_RANKING.json](configs/RECSYS_RANKING.json)|
|RECSYS_RANKING_GR_2B|[RECSYS_RANKING_GR_2B.json](configs/RECSYS_RANKING_GR_2B.json)|
|RECSYS_RANKING_GR_7B|[RECSYS_RANKING_GR_7B.json](configs/RECSYS_RANKING_GR_7B.json)|
|RECSYS_RETRIEVAL|[RECSYS_RETRIEVAL.json](configs/RECSYS_RETRIEVAL.json)|
|SASREC_1M|[SASREC_1M.json](configs/SASREC_1M.json)|
|SASREC_20M|[SASREC_20M.json](configs/SASREC_20M.json)|
|SharedBottom|[SharedBottom.json](configs/SharedBottom.json)|
|SIM|[SIM.json](configs/SIM.json)|
|TDM|[TDM.json](configs/TDM.json)|
|WideDeep|[WideDeep.json](configs/WideDeep.json)|
|WideandDeep|[WideandDeep.json](configs/WideandDeep.json)|
|Yolov5|[Yolov5.json](configs/Yolov5.json)|
|dlrmHSTU|[dlrmHSTU.json](configs/dlrmHSTU.json)|
|wukong|[wukong.json](configs/wukong.json)|
|xDeepFM|[xDeepFM.json](configs/xDeepFM.json)|

# 性能指标

模型正常运行后，会在models目录下生成性能相关文件，目录为./models/save\_results\_{device\_name}/performance\_result.txt,其中{device\_name}为运行设备名称，如npu、cuda、cpu,文件内容为模型的性能指标，如推理时间、qps等。
如config中开启了profiling\_flag，会在./models/profiling/{model\_name}生成profiling结果文件。其中{model\_name}为模型名字。

# 精度指标

模型正常运行后，会在models目录下生成落盘输出文件，目录为save\_results\_{device\_name}/{model\_name},其中device\_name为运行设备名称，如npu、cuda、cpu,{model\_name}为模型名字。

使用tools目录下的脚本对两份数据进行对比，cpu的数据作为标杆，对比npu数据与cpu数据的差异。
对比脚本为tools/compare\_output.py，使用方法为：

```shell
python ./tools/compare_output.py --actual_output ./models/save_results_npu --expected_output ./models/save_results_cpu  --rtol 1e-4 --atol 1e-4
```

其中--actual\_output为npu数据目录，--expected\_output为cpu数据目录，--rtol为相对误差容忍度，--atol为绝对误差容忍度。

# config文件示例

```json
{
    "name": "AutoInt",
    "url": "https://github.com/reczoo/FuxiCTR.git",
    "commit_id": "b7dff736885fdb8f59387d82d08219ad2e4cae50",
    "patch_path": "patches/fuxictr_npu.patch",
    "type": "infer",
    "epoch": 100,
    "profiling_flag": true,
    "aclgraph_flag": true,
    "data_type": "float32",
    "run_cmd": [
        "python",
        "model_zoo/DMR/run_expid.py",
        "--expid",
        "DMR_test",
        "--gpu",
        "0"
    ],
    "pip_install_self": true,
    "pip_install_requirements": true,
    "extra_cmd": ["pip install Scikit-learn<1.5"]
}
```

- name:模型名字
- url:模型代码仓下载路径
- commit\_id:本示例适配的commit节点
- patch\_path:适配的patch目录
- type：推理还是训练模式(infer/train/train\_evaluate)
- epoch: 训练步数或者推理循环次数
- profiling\_flag: 是否抓取profiling
- aclgraph\_flag: 是否需要使能图下沉
- data\_type: 模型的input数据类型(float32/float16/bfloat16)
- run\_cmd: 模型的运行命令
- pip\_install\_self: 是否依赖安装开源仓自己
- pip\_install\_requirements: 开源仓是否要安装其根目录下的requirements.txt依赖包
- extra\_cmd: 适配npu需要额外执行的命令

# 模型额外操作说明

## yolov5 模型

yolov5模型需手动下载权重文件。<https://gitcode.com/open-source-toolkit/6e474/blob/main/yolov5%20%E5%AE%98%E6%96%B9%E6%9D%83%E9%87%8D%E6%96%87%E4%BB%B6.zip>
从链接里下载并解压，把yolov5s.pt放在ckpt文件夹下再运行。

## Multitask-Recommendation-Library(MMOE) 模型

MMOE模型运行前需访问 <https://tianchi.aliyun.com/dataset/74690> 下载aliexpress\_NL\_datasets.zip数据集，
并将数据集放到**与本README.md同级的data目录**下。

## DLRM\_META 模型

[DLRM\_META](https://github.com/facebookresearch/dlrm.git)模型运行需要下载[Kaggle Display Advertising dataset](https://ailab.criteo.com/ressources/)数据集,
在benchmark/datasets目录下新建dlrm\_meta目录存放原始train.txt和test.txt文件。

```shell
|-- benchmark
   |-- configs
       |-- DLRM_META.json
   |-- datasets
       |-- dlrm_meta
           |-- train.txt
           |-- test.txt
```

GPU运行前，需要将DLRM_META.json的`--use-npu`选项替换为`--use-gpu`选项

## HSTU\_META 模型

### 运行依赖

- RecSDK自定义算子安装: 参考[README](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md), 包含下列自定义算子:
  - asynchronous\_complete\_cumsum
  - dense\_to\_jagged
  - hstu\_dense\_backward
  - hstu\_dense\_forward
  - jagged\_to\_padded\_dense
  - invert\_permute
  - permute2d\_sparse\_data

### 准备数据集

[HSTU\_META](https://github.com/meta-recsys/generative-recommenders)数据集自动下载，如果下载失败可以参考开源代码处理。

## NV Recsys-examples 模型

- 包含开源gr_ranking、gr_retrival
- 包含增加MoE结构的GR RANKING 7B
- 包含增加GroupedMatmul优化的GR RANKING 2B

### 运行依赖

- RecSDK自定义算子安装: 参考[README](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md), 包含下列自定义算子:
  - asynchronous\_complete\_cumsum
  - dense\_to\_jagged
  - hstu\_dense\_backward
  - hstu\_dense\_forward
  - jagged\_to\_padded\_dense
  - invert\_permute
  - permute2d\_sparse\_data
- [DynamicEmbedding for NPU](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v2/dynamic_emb/README.md): 使用源码方式安装
- [Torchrec NPU for Recsys-example](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v2/torchrec_npu/README.md)

### 下载指定版本的训练套件(脚本自动下载，可跳过)

训练套件依赖:

- Megatron-LM (core\_r0.14.0)
- MindSpeed(core\_r0.14.0)

在recsys-example目录下载Mindspeed文件夹和Megatron-LM文件夹

```shell
|-- recsys-example
   |-- Mindspeed
   |-- Megatron-LM
   |-- recsys-examples-NV

# Megatron
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.14.0

# MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git 
cd MindSpeed
git checkout core_r0.14.0
```

### 准备数据集

[NV Recsys-examples](https://github.com/NVIDIA/recsys-examples)模型运行前需要按[步骤](https://github.com/NVIDIA/recsys-examples/blob/v25.09/examples/hstu/README.md#dataset-preprocessing)准备movielen-20m数据集

在benchmark/datasets目录新建recsys\-examples目录存放处理好的ml-20m数据集目录。

```shell
|-- benchmark
   |-- configs
       |-- RECSYS_RANKING.json
       |-- RECSYS_RETRIEVAL.json
   |-- datasets
       |-- recsys-examples
           |-- ml-20m
```

## SASRec 模型

### 运行依赖

- 本模型依赖以下fbgemm算子。若需在昇腾NPU环境部署运行，请先安装适配包fbgemm_ascend，安装及使用说明请参考官方[README](https://gitcode.com/Ascend/fbgemm-ascend/blob/v1.5.0/README.md)。
  - asynchronous\_complete\_cumsum
  - dense\_to\_jagged
  - jagged\_to\_padded\_dense

### 准备数据集

[SASRec](https://github.com/meta-recsys/generative-recommenders)数据集自动下载，如果下载失败可以参考开源代码处理。

## GRU4Rec 模型

### 准备数据集

[GRU4Rec](https://github.com/hidasib/GRU4Rec_PyTorch_Official)数据集自动下载解析，如下载失败可以参考开源代码下载RetailRocket数据集，然后在模型目录下使用`python retailrocket_preproc.py -p ./data`命令预处理

## RANKMIXER 模型

### 准备数据集

amazon books数据集处理参考开源代码[HSTU\_META](https://github.com/meta-recsys/generative-recommenders)，下载HSTU源码后，执行`mkdir -p tmp/ && python3 preprocess_public_data.py`

处理完成后，将HSTU目录下的`generative-recommenders/tmp/amzn_books/sasrec_format.csv`放到本项目的`benchmark/datasets`目录下即可

```shell
|-- benchmark
   |-- configs
       |-- RANKMIXER.json
   |-- datasets
       |-- rankmixer
           |-- sasrec_format.csv
```
