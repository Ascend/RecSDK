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
|DECISION_TRANSFORMER|[DecisionTransformer.json](configs/DecisionTransformer.json)|
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
|EasyRec|[EASYREC.json](configs/EASYREC.json)|
|ESMM|[ESMM.json](configs/ESMM.json)|
|ESMM_TRAIN|[ESMM_TRAIN.json](configs/ESMM_TRAIN.json)|
|ETA|[ETA.json](configs/ETA.json)|
|EulerNet|[EulerNet.json](configs/EulerNet.json)|
|FiBiNET|[FiBiNET.json](configs/FiBiNET.json)|
|GRU4Rec|[GRU4Rec.json](configs/GRU4Rec.json)|
|HSTU_META_1M|[HSTU_META_1M.json](configs/HSTU_META_1M.json)|
|HSTU_META_1M_LARGE|[HSTU_META_1M_LARGE.json](configs/HSTU_META_1M_LARGE.json)|
|HSTU_META_20M|[HSTU_META_20M.json](configs/HSTU_META_20M.json)|
|HSTU_META_20M_LARGE|[HSTU_META_20M_LARGE.json](configs/HSTU_META_20M_LARGE.json)|
|ULTRA_HSTU_META_1M_NPU|[ULTRA_HSTU_META_1M_NPU.json](configs/ULTRA_HSTU_META_1M_NPU.json)|
|ULTRA_HSTU_META_1M_GPU|[ULTRA_HSTU_META_1M_GPU.json](configs/ULTRA_HSTU_META_1M_GPU.json)|
|HSTU_META_7B|[HSTU_META_7B.json](configs/HSTU_META_7B.json)|
|IFM|[IFM.json](configs/IFM.json)|
|MIND|[MIND.json](configs/MIND.json)|
|MMOE|[MMOE.json](configs/MMOE.json)|
|MaskNet|[MaskNet.json](configs/MaskNet.json)|
|MultiTower|[MultiTower.json](configs/MultiTower.json)|
|Multitask-Recommendation-Library|[Multitask-Recommendation-Library.json](configs/Multitask-Recommendation-Library.json)|
|NCF|[NCF.json](configs/NCF.json)|
|NFM|[NFM.json](configs/NFM.json)|
|ONN|[ONN.json](configs/ONN.json)|
|OpenP5|[OpenP5.json](configs/OpenP5.json)|
|OpenP5_LLaMA|[OpenP5_LLaMA.json](configs/OpenP5_LLaMA.json)|
|PLE|[PLE.json](configs/PLE.json)|
|PNN|[PNN.json](configs/PNN.json)|
|RANKMIXER|[RANKMIXER.json](configs/RANKMIXER.json)|
|TOKENMIXER_LARGE|[TOKENMIXER_LARGE.json](configs/TOKENMIXER_LARGE.json)|
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

## ETA 与 ESMM\_TRAIN 模型

ETA和ESMM\_TRAIN使用共用预处理后的[Ali-CCP数据集](https://tianchi.aliyun.com/dataset/408)，默认从`benchmark/dataset/aliccp_out`读取。

ETA通过`MODEL_TYPE`控制模型数据类型（默认`float32`,可选`bfloat16`），通过`EMB_TYPE`控制Embedding数据类型（默认`model`，即跟随模型数据类型）。ESMM\_TRAIN默认配置为`BS=32`、`DIM=8`、`TRAIN_STEP=10240`和`EVAL_STEP=1`。

两个配置均使用训练框架收集的全部有效`train_times`统计性能。`RESULTS_DIR`使用rec-models默认值，ETA和ESMM\_TRAIN的结果分别保存到`benchmark/models/eta/save_results_NPU`和`benchmark/models/esmm/save_results_NPU`。

## Decision Transformer 模型

### 精度模式

Decision Transformer 默认关闭精度模式。进行精度比对时，可将 `DecisionTransformer.json` 中的 `P_TEST` 设置为 `1`。开启后会调用 msProbe 的 `seed_all` 接口，固定随机种子并关闭 dropout；同时会固定 Linear 和 Embedding 参数，关闭训练数据随机采样，并设置 NPU、HCCL 的确定性环境变量。

使用精度模式前，需要在运行模型的 Python 环境中安装 msProbe：

```shell
pip install mindstudio-probe
```

安装完成后，可执行 `pip show mindstudio-probe` 检查是否安装成功。其他安装方式及版本配套信息请参考 [msProbe 工具安装指南](https://gitcode.com/Ascend/msprobe/blob/26.0.0/docs/zh/msprobe_install_guide.md)。

### 准备数据集

模型默认从 `benchmark/dataset/decision_transformer/training_data_all-rlData.csv` 读取离线强化学习轨迹。CSV 文件需要包含 `state`、`next_state`、`action`、`reward` 和 `done` 字段，其中 `state` 和 `next_state` 为长度为 16 的数组，`done=True` 表示当前轨迹结束。

```shell
|-- benchmark
   |-- configs
       |-- DecisionTransformer.json
   |-- dataset
       |-- decision_transformer
           |-- training_data_all-rlData.csv
```

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

DLRM_META启用Inductor模式时，需要在DLRM_META.json的`run_cmd`中添加`--no-use-emb-sparse`选项，以禁用Embedding稀疏梯度。

## HSTU\_META 模型

### 运行依赖

HSTU NPU 适配补丁通过 Python `import` 导入依赖库并注册自定义算子，不需要在模型代码中手动加载算子 `.so` 文件。运行前请确保当前 Python 环境已安装与 PyTorch、CANN 版本配套的以下依赖，且可以正常导入：

| 依赖库 | 导入方式 | 用途 |
|--|--|--|
| `fbgemm_ascend` | `import fbgemm_ascend` | 注册 `torch.ops.fbgemm` 下的 Jagged Tensor 等算子 |
| `rec_cust_ops` | `import rec_cust_ops` | 注册 `torch.ops.mxrec` 下的 RecSDK 自定义算子 |
| `ops_rec` | `import ops_rec` | 提供 `ops_rec.ascendc.attention.hstu_jagged` 等 HSTU 融合算子 |

`fbgemm_ascend` 安装说明请参考 [fbgemm-ascend README](https://gitcode.com/Ascend/fbgemm-ascend/blob/v1.5.0/README.md)，`rec_cust_ops` 编译安装说明请参考 [RecSDK 自定义算子 README](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)。

#### 精度模式依赖

HSTU 模型开启精度模式（将对应配置文件中的 `ENABLE_PRECISION_MODE` 设置为 `1`）时，会调用 msProbe 提供的 `seed_all` 接口，并关闭训练过程中的 dropout，以降低随机性对精度比对的影响。使用该模式前，需要在运行模型的 Python 环境中额外安装 msProbe：

同时开启 RAB（`ENABLE_RAB=1`）和精度模式时，为保证确定性，RAB 中的 `index_select` 会采用固定顺序执行，因此训练速度相比非精度模式会有所下降。

```shell
pip install mindstudio-probe
```

安装完成后，可执行 `pip show mindstudio-probe` 检查是否安装成功。其他安装方式及版本配套信息请参考 [msProbe 工具安装指南](https://gitcode.com/Ascend/msprobe/blob/26.0.0/docs/zh/msprobe_install_guide.md)。

### 准备数据集

运行 [HSTU\_META](https://github.com/meta-recsys/generative-recommenders) 时，配置会优先检查 `benchmark/dataset/hstu_meta` 目录。如果该目录中已有数据，则将其复制到开源仓的 `tmp/` 目录；如果该目录为空，则执行 `preprocess_public_data.py` 自动下载并预处理数据。

在无网络环境中，请提前将准备好的数据放入 `benchmark/dataset/hstu_meta` 目录，目录结构如下：

```shell
|-- benchmark
   |-- dataset
       |-- hstu_meta
           |-- ml-20m
           |-- ml-1m
           |-- processed
```

### 设备配置

运行 HSTU 模型前，应根据实际设备类型修改对应配置文件 `run_cmd` 中的设备可见性环境变量：

- NPU 环境修改 `ASCEND_RT_VISIBLE_DEVICES`。
- GPU 环境修改 `CUDA_VISIBLE_DEVICES`。

例如，单卡运行时将对应变量设置为 `0`，8 卡运行时设置为 `0,1,2,3,4,5,6,7`。HSTU 配置文件同时保留了这两个变量，实际运行时只需修改与当前设备对应的变量；不要使用 `CUDA_VISIBLE_DEVICES` 配置 NPU，也不要使用 `ASCEND_RT_VISIBLE_DEVICES` 配置 GPU。

### 训练控制参数

- `--eval_every_n N`：每隔 `N` 个 epoch 执行一次 epoch 评估，`N` 必须为正整数；当前 HSTU 配置默认设为 `50`，并且训练的最后一个 epoch 仍会执行评估。
- `STOP_STEP=N`：训练达到 `N` 个 step 后提前停止，设置为 `0` 时关闭；当前 `HSTU_META_7B.json` 默认设为 `200`。提前停止时不执行 epoch 评估，也不输出 `metrics` 字段。

## NV Recsys-examples 模型

- 包含开源gr_ranking、gr_retrieval
- 包含增加MoE结构的GR RANKING 7B
- 包含增加GroupedMatmul优化的GR RANKING 2B

### 运行依赖

NV Recsys-examples 的 NPU 适配补丁通过 Python `import` 导入依赖库并注册自定义算子，不需要在模型代码中手动加载算子 `.so` 文件。请根据所运行的模型，安装与 PyTorch、CANN 版本配套的依赖库：

| 模型 | 依赖库 |
|--|--|
| GR Ranking、GR Ranking 2B、GR Ranking 7B | `fbgemm_ascend`、`rec_cust_ops`、`ops_rec` |
| GR Retrieval | `fbgemm_ascend`、`ops_rec` |

各依赖库的导入方式和用途与上文 HSTU_META 模型的运行依赖说明相同。

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

[NV Recsys-examples](https://github.com/NVIDIA/recsys-examples)模型运行前需要按[步骤](https://github.com/NVIDIA/recsys-examples/blob/v25.09/examples/hstu/README.md#dataset-preprocessing)准备MovieLens 20M数据集

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

SASRec 在昇腾 NPU 环境中通过 `import fbgemm_ascend` 注册 `torch.ops.fbgemm` 下的自定义算子，不需要手动加载算子 `.so` 文件。运行前请安装与 PyTorch、CANN 版本配套的 `fbgemm_ascend` 依赖库，安装及使用说明请参考 [fbgemm-ascend README](https://gitcode.com/Ascend/fbgemm-ascend/blob/v1.5.0/README.md)。

### 准备数据集

[SASRec](https://github.com/meta-recsys/generative-recommenders)数据集自动下载，如果下载失败可以参考开源代码处理。

## GRU4Rec 模型

### 准备数据集

[GRU4Rec](https://github.com/hidasib/GRU4Rec_PyTorch_Official)数据集自动下载解析，如下载失败可以参考开源代码下载RetailRocket数据集，然后在模型目录下使用`python retailrocket_preproc.py -p ./data`命令预处理

## RANKMIXER 模型

### 精度模式

RANKMIXER 默认关闭精度模式。进行 GPU 与 NPU 精度比对时，可将 `RANKMIXER.json` 中的 `ENABLE_PRECISION_MODE` 设置为 `1`。开启后，会使用 msProbe 的 `seed_all` 接口，使用 `seed_conf.global_seed` 固定随机数（默认值为 `1234`），并关闭 dropout。同时，开启精度模式也会固定模型的初始化权重。

使用精度模式前，需要在运行模型的 Python 环境中安装 msProbe：

```shell
pip install mindstudio-probe
```

安装完成后，可执行 `pip show mindstudio-probe` 检查是否安装成功。其他安装方式及版本配套信息请参考 [msProbe 工具安装指南](https://gitcode.com/Ascend/msprobe/blob/26.0.0/docs/zh/msprobe_install_guide.md)。

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

## OpenP5 模型

### 适配范围

- 上游仓库：<https://github.com/agiresearch/OpenP5>
- 固定 commit：`7f110389cd5ab51820e29e94a44b6db83df243fb`
- 适配 patch：`patches/openp5_npu.patch`
- 配置文件：`configs/OpenP5.json`
- 模型：`t5-small` revision
  `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`
- 数据集：OpenP5 官方 ML-1M
- 任务：`sequential,straightforward`

Patch 新增 CUDA/torch_npu benchmark 入口、单卡与 HCCL/DDP 多卡执行、固定
checksum 的资产准备、逐 optimizer-step loss 和同步时延统计。

### 环境与依赖

已验证环境为 Atlas 800T A3（`Ascend910_9382`）、CANN 9.0、Python 3.11.15、
PyTorch 2.7.1 和 torch_npu 2.7.1。请先安装与 CANN 匹配的 PyTorch/torch_npu；
以下命令不会安装或替换 PyTorch、torch_npu、CANN 和驱动：

```shell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  transformers==4.26.0 safetensors==0.8.0 sentencepiece==0.2.2 \
  scikit-learn==1.7.2 scipy==1.15.3
```

### RecSDK 默认启动

在 `benchmark/` 目录执行：

```shell
python run.py OpenP5.json --eager
```

`OpenP5.json` 会完成以下操作：

1. clone OpenP5 并 checkout 固定 commit；
2. 应用 `patches/openp5_npu.patch`；
3. 执行 `python adaptation/prepare_assets.py` 准备固定版本的 T5-small 和 ML-1M；
4. 在逻辑设备 `0..7` 启动 T5 BF16 8P 性能 workload。

默认 workload 的 global batch 为 128，每 rank microbatch 为 16，gradient
accumulation 为 1，共运行 220 个 optimizer steps；排除 steps 1..20 后统计
steps 21..220。运行前须通过 `npu-smi info` 确认八张卡空闲。如物理设备编号不是
`0..7`，先设置可见设备，将选中的八张物理卡映射为逻辑设备 `0..7`：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

受限网络可在运行 `run.py` 前设置任务环境允许的 HTTP/HTTPS proxy，并使用：

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

离线环境可准备以下目录，并在模型目录中导入：

```text
<asset-root>/
|-- t5-small/
`-- data/ML1M/
```

```shell
python adaptation/prepare_assets.py --source-root <asset-root>
```

### T5 三精度单卡复现

在 `benchmark/models/OpenP5/` 目录中，选择一张空闲物理 NPU 并映射为逻辑
`npu:0`：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_t5_accuracy() {
  precision=$1
  learning_rate=$2
  output=$3
  python src/src_t5/benchmark_npu.py \
    --device npu --device_ids 0 --precision "$precision" \
    --backbone adaptation/assets/t5-small \
    --datasets ML1M --tasks sequential,straightforward \
    --item_indexing sequential --data_path adaptation/assets/data \
    --prompt_file prompt.txt --sample_prompt 1 --sample_num 1,1 --max_his 20 \
    --global_batch_size 8 --gradient_accumulation_steps 1 \
    --max_optimizer_steps 1000 --scheduler_steps 1000 --warmup_prop 0.05 \
    --measurement_warmup_steps 20 --grad_scaler_init_scale 1.0 \
    --optimizer sgd --sgd_momentum 0 --lr "$learning_rate" --weight_decay 0.01 \
    --clip 1 --dropout 0.0 --random_initialize 1 --logging_step 200 \
    --output_dir "$output" --run_name "a3_t5_accuracy_${precision}"
}

run_t5_accuracy float32 0.0001 adaptation/evidence/t5_accuracy_npu_float32
run_t5_accuracy float16 0.00001 adaptation/evidence/t5_accuracy_npu_float16
run_t5_accuracy bfloat16 0.0001 adaptation/evidence/t5_accuracy_npu_bfloat16
```

H100 对照使用相同参数，仅把 `--device npu` 改为 `--device cuda` 并设置
`CUDA_VISIBLE_DEVICES`。两端固定源码、checkpoint/init、seed、样本顺序、global
batch、SGD、scheduler 和 gradient accumulation；每行 `steps.jsonl` 对应一次完整
optimizer update，不是 microstep。精度验收采用无动量 SGD，避免 AdamW 二阶矩归一化
放大设备间的梯度舍入差异；默认性能 workload 仍使用 AdamW。正常的 50-step warmup
后进入衰减段，不使用大 epsilon 或全程 warmup 抑制参数更新。FP16 如发生 overflow
会立即终止，避免把跳过更新的 microstep 计作 optimizer step；本次两端均完成 1000 步。

|精度|1000-step loss MAE|严格门槛|最大绝对误差|结果|
|--|--:|--:|--:|--|
|FP32|`2.8099537e-5`|`<1e-4`|`2.1076202e-4`（step 864）|PASS|
|FP16|`7.9107475e-4`|`<1e-3`|`2.7289391e-2`（step 45）|PASS|
|BF16|`6.2802196e-3`|`<1e-2`|`1.2673664e-1`（step 11）|PASS|

### T5 性能结果

性能为 BF16、global batch 128；每个 optimizer step 前后同步设备。计时包含
H2D、forward、backward、gradient clipping、AdamW 和 scheduler，不包含 CPU
dataset lookup/tokenization；8P 取所有 rank 的最大时间。

|模式|World size|每 rank microbatch|Mean ms/step|Median ms/step|Stddev ms|P99 ms|P999 ms|
|--|--:|--:|--:|--:|--:|--:|--:|
|H100 1P|1|128|59.471|61.551|2.992|65.637|67.448|
|A3 1P|1|128|78.967|85.755|18.803|99.304|99.305|
|A3 8P|8|16|100.206|99.367|14.125|123.851|138.268|

固定 global batch 下，A3 8P 比 A3 1P 慢约 26.9%。T5 每 rank microbatch 降为
16 后，计算量不足以覆盖 HCCL all-reduce、同步和最慢 rank 尾延迟，因此不宣称
当前 workload 具有 8P 加速。

### 输出说明

每个 benchmark 输出目录包含：

- `manifest.json`：设备、精度、batch 和测量窗口；
- `steps.jsonl`：逐 optimizer-step loss、同步时延和输入 hash；
- `summary.json`：排除 warmup 后的 mean、median、stddev、P90、P95、P99 和
  P999。

同时会将标准性能指标追加到
`benchmark/models/save_results_<device>/performance_result.txt`。如需采集
profiling，将 `configs/OpenP5.json` 中的 `profiling_flag` 改为 `true` 后单独运行；
结果保存在 `benchmark/models/profiling/OpenP5/<device>/`，该次运行不用于正式
性能对比。

数值验收前必须先核对 GPU/NPU manifest 和每步 input hash，再按显式
`optimizer_step` 计算前 1000 步：

```text
MAE = mean(abs(npu_loss[i] - gpu_loss[i]))
```

## OpenP5 OpenLLaMA 模型

本配置适配 OpenP5 的 `openlm-research/open_llama_3b_v2` LoRA 训练路径，固定 OpenP5 commit `7f110389cd5ab51820e29e94a44b6db83df243fb`，
使用 ML-1M 的 `sequential,straightforward` 任务。Patch 为 `patches/openp5_llama_npu.patch`，配置为 `configs/OpenP5_LLaMA.json`。

### 环境与默认启动

NPU 环境需预先安装匹配 CANN 的 PyTorch/torch_npu。OpenLLaMA 依赖安装在项目
overlay 中，不替换系统 PyTorch：

```shell
mkdir -p adaptation/deps
python -m pip install --target adaptation/deps --no-deps \
  transformers==4.31.0 peft==0.5.0 accelerate==0.25.0
```

在 `benchmark/` 目录执行默认 BF16 A3 八卡、global batch 8、220 optimizer
steps 性能 workload：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python run.py OpenP5_LLaMA.json --eager
```

配置会自动应用 patch、准备固定模型/ML-1M，并限制每 rank CPU 线程数为 1。

### 三精度单卡复现

`OpenP5_LLaMA.json` 是使用 AdamW 的 BF16 八卡性能配置。精度复现两端改用无动量 SGD，避免二阶矩归一化放大梯度舍入差异；
scheduler 为 1000 steps、warmup 为 50 steps，FP32 global batch 为 1，FP16/BF16 为 8。在模型目录将一张物理 NPU 映射为逻辑设备 0：

```shell
export PYTHONPATH=$PWD/adaptation/deps
export RECSDK_BENCHMARK_MODELS_ROOT=..
run_llama_accuracy() {
  precision=$1
  global_batch=$2
  learning_rate=$3
  python src/src_llama/benchmark_llama_npu.py \
    --device npu --device_ids 0 --precision "$precision" \
    --model_path adaptation/assets/open_llama_3b_v2 \
    --data_path adaptation/assets/data --dataset ML1M \
    --tasks sequential,straightforward --prompt_file prompt.txt \
    --max_his 10 --cutoff 512 --seed 2023 \
    --global_batch_size "$global_batch" --gradient_accumulation_steps 1 \
    --max_optimizer_steps 1000 --scheduler_steps 1000 \
    --scheduler_warmup_steps 50 --measurement_warmup_steps 20 \
    --optimizer sgd --sgd_momentum 0 --learning_rate "$learning_rate" --weight_decay 0.01 \
    --clip 1 --grad_scaler_init_scale 1.0 --lora_r 8 --lora_alpha 16 \
    --lora_targets q_proj,v_proj,embed_tokens --logging_step 250 \
    --output_dir "adaptation/evidence/llama_${precision}" \
    --run_name "llama_${precision}"
}
run_llama_accuracy float32 1 0.0001
run_llama_accuracy float16 8 0.00001
run_llama_accuracy bfloat16 8 0.00001
```

H100 使用相同参数，仅将 `--device npu` 改为 `--device cuda`。两端均在 50-step
warmup 后进入衰减段，未使用大 epsilon 或全程 warmup 抑制参数更新。前 1000 个
optimizer steps 的验收结果：

|精度|Global batch|Loss MAE|验收阈值|结果|
|--|--:|--:|--:|--|
|FP32|1|`4.2591095e-6`|`<1e-4`|PASS|
|FP16|8|`7.7148390e-4`|`<1e-3`|PASS|
|BF16|8|`6.4460926e-3`|`<1e-2`|PASS|

### 性能与 profiling

BF16 性能固定 global batch 8，排除前 20 个 optimizer steps，统计 steps
21..220。延迟是包含 H2D、前向、反向、梯度裁剪、AdamW 和 scheduler 的完整
optimizer step，8P 取各 rank 最大值；吞吐率为 global batch 除以 mean step 时间，
单位为训练 samples/s。结果包含 mean、median、stddev、P90、P95、P99、P999，并
写入 `benchmark/models/save_results_<device>/performance_result.txt`。

|模式|Mean ms|Median ms|Stddev ms|P99 ms|P999 ms|
|--|--:|--:|--:|--:|--:|
|H100 1P|1441.840|1437.499|30.916|1588.061|1588.879|
|A3 1P|1149.115|1146.479|18.267|1197.807|1218.439|
|A3 8P|196.476|195.115|11.504|223.123|226.417|

将 `OpenP5_LLaMA.json` 的 `profiling_flag` 改为 `true` 可独立采集 NPU profiler，
结果保存到 `benchmark/models/profiling/OpenP5_LLaMA/npu/`；profiling 运行不用于
正式性能对比。

## EasyRec 模型

运行前准备与 Atlas A2/A3、CANN 和 Python 匹配的 PyTorch/torch_npu，并安装 `uv`、`wget`、`unzip`；配置会自动准备其余依赖、官方 checkpoint 和数据。

默认八卡 bf16 性能运行，单卡将 `NPROC_PER_NODE` 设为 `1`：

```shell
python run.py EASYREC.json --eager
NPROC_PER_NODE=1 python run.py EASYREC.json --eager
```

精度对比使用以下确定性配置（batch size 1、学习率 `5e-6`、1000 次更新）：

```shell
NPROC_PER_NODE=1 EASYREC_PRECISION=float32 EASYREC_STEPS=1000 \
EASYREC_BATCH_SIZE=1 EASYREC_LEARNING_RATE=5e-6 EASYREC_DETERMINISTIC=1 \
python run.py EASYREC.json --eager --no_hf32
```

`EASYREC_PRECISION` 支持三种精度；默认性能运行使用 batch size 32、`5e-5` 学习率和非确定性 dropout。loss、性能摘要分别写入 `models/save_results_npu/EasyRec/` 和 `models/save_results_npu/performance_result.txt`；`--cpu`、`--eager`、`--no_hf32` 与 `profiling_flag` 均可按通用 benchmark 方式使用。
