# rec_model_zoo_pytorch

该文档介绍如何执行模型仓推理流程

## 主要依赖

Pytorch==2.7.1
torch_npu==2.7.1
tokenization==1.0.7
opencv-python-headless==4.11.0.86
deepctr_torch==0.2.9
triton_ascend
easydict

### 虚拟数据集

设置MODE=test_qps，会使用生成的虚拟数据集进行推理

### SASRec数据集m1-lm

sasrec模型数据集从[该路径](https://github.com/pmixer/SASRec.pytorch/blob/main/python/data/ml-1m.txt)获取，然后放入数据集路径下：`rec_model_zoo_pytorch/datasets/ml-1m/`
如果没有ml-1m文件夹需要自行新建一个

## 模型训推

首先需要在 `launch.sh`中设置参数:

| 参数名称             | 参数说明                                  |
| -------------------- | ----------------------------------------- |
| JOB_ID               | 任务id ｜                                 |
| PREPROCESSED_DATASET | 数据集路径                                |
| DEVICE               | 运行设备名称，目前支持cpu/cuda/npu        |
| DEVICE_ID            | 设备id序号                                |
| OPEN_PROFILING       | 打开profiling记录设备运行情况             |
| PROFILING_PATH       | 设置profiling文件输出路径                 |
| MODE                 | train/infer/eval 设置训练、推理、验证模式 |
| models               | 设置需要运行的模型，支持模型见下方列表    |

| 类型                     | 模型     |
| ------------------------ | -------- |
| feature interaction      | DLRM     |
| feature interaction      | AutoInt  |
| feature interaction      | DCN_v2   |
| feature interaction      | DLRM     |
| behaviour and multi-task | ETA      |
| behaviour and multi-task | MMOE     |
| behaviour and multi-task | DIEN     |
| behaviour and multi-task | ESMM     |
| behaviour and multi-task | SASRec   |
| ByteMLPerf               | Resnet50 |
| ByteMLPerf               | Bert     |

1. 对于**feature interaction learning**模型:

   ```shell
   cd feature_interaction/
   bash launch.sh
   ```
2. 对于 **behaviour sequence modeling** 和 **multi-task learning** 模型:

   ```shell
   cd behaviour_and_multi_task/
   bash launch.sh
   ```
3. 对于 **ByteMLPerf** 路径下的模型:
   需要在Huggingface下载原始模型配置文件：

| 模型 | 路径 |
| ---- | ---- |
| Resnet50 | https://huggingface.co/google-bert/bert-base-uncased |
| Bert | https://huggingface.co/microsoft/resnet-50 |

然后进入bytemlperf文件夹并修改launch.sh脚本中`MODEL_PATH`为模型路径，模型路径结构如下：
```
├── bert-torch
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
└── resnet50-torch-fp32
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin
```
最后执行`bash launch.sh`进行推理。

## Profiling

1. 在launch.sh中启用Profiling并设置profiling路径:
   ```shell
   OPEN_PROFILING=true
   PROFILING_PATH=${cur_path}/profiling
   ```
2. 运行 `launch.sh`脚本，以特征交互模型为例:
   ```shell
   cd feature_interaction/src/
   bash launch.sh
   ```
3. 将`PROFILING_PATH`路径下将输出的trace*.json文件拖入chrome://tracing页面即可查看运行时详细信息
