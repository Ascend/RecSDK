# HLLM模型迁移样例说明

## 适配说明

本样例以开源HLLM模型为基准，将其迁移适配到NPU进行训练。原模型参考：[开源代码链接](https://github.com/bytedance/HLLM/blob/main/README.md)。 

## 代码结构说明

```shell
├── configs
│   ├── HLLM.json             # 模型运行配置 
├── patchs
│   ├── hllm_npu.patch        # 模型迁移适配patch文件
└── models
    ├── launch_npu.sh             # 模型运行脚本
    ├── README.md                 # 样例迁移说明文档
    ├── hllm_gpu_precision.patch  # 模型精度对齐patch文件
```

## 运行环境准备

本模型迁移仅支持昇腾950DT系列产品以及配套的CANN版本，推荐Python版本为3.11.X，Pytorch版本为2.7.1配套。

| 依赖      | 版本      | 
|-----------|-----------|
| torch     | 2.7.1+cpu |
| torch-npu | 2.7.1     | 

## 数据集准备

在/benchmark/dataset/目录下创建名为hllm的目录存放训练数据集和预训练模型参数。

### 下载数据集

1.参考 [开源代码链接](https://github.com/bytedance/HLLM/blob/main/README.md) 准备Amazon Book Reviews数据集，通过 `Interactions` 链接下载的文件重命名为`amazon_books.csv`放入"hllm/dataset/"目录中，通过 `Item Information` 链接下载的文件重命名为`amazon_books.csv`放入"hllm/information/"目录中

### 下载预训练模型

1.下载[TinyLlama 1B](https://hf-mirror.com/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/tree/main) 预训练模型，放入"hllm/pretrain/"目录中

```shell
├── hllm     # hllm训练数据集目录 
│   ├── dataset
│       ├── amazon_books.csv
│   ├── information
│       ├── amazon_books.csv
│   ├── pretrain
│       ├── TinyLlama

```

## 模型运行

在benchmark目录下执行以下命令模型启动

```shell
python3 run.py HLLM.json --eager
```

## 主要模型参数介绍

1.进入适配后的源码文件`HLLM/code/HLLM/HLLM.yaml`和`HLLM/code/overall/LLM_deepspeed_npu.yaml`确认参数。值得关注的参数如下：

```shell
# LLM_deepspeed_npu.yaml
data_path         # 数据集路径，如按以上操作默认在../dataset/
text_path         # information数据集路径，如按以上操作默认在../information/
dataset           # 数据集名称，默认amazon_books
epochs            # 训练轮次数，默认1
train_batch_size  # batch_size大小，默认1

# HLLM.yaml
item_pretrain_dir # 预训练模型路径，如按以上操作默认在../pretrain/TinyLlama，user_pretrain_dir同理。

```

2.启动脚本参数可参考`launch_npu.sh`文件

```shell
DEVICE_ID         # 设置设备ID
MASTER_PORT       # 设置端口号
ENABLE_PROFILER   # 是否开启profiling，默认关闭
TRAIN_STOP_STEP   # early_stop步数，默认是1000
```

## 双机多卡运行

```shell
# 机器0的配置修改
DEVICE_ID=0,1,2,3,4,5,6,7   
MASTER_ADDR="xxx.xxx.x.xxx" # 机器0的IP
MASTER_PORT=12345           
NODE_RANK=0                 # 机器0 
NNODES=2                    # 双机

# 机器1的配置修改
DEVICE_ID=0,1,2,3,4,5,6,7   
MASTER_ADDR="xxx.xxx.x.xxx" # 机器0的IP(与机器0保持一致)
MASTER_PORT=12345
NODE_RANK=1                 # 机器1
NNODES=2                    # 双机
```

注意：启动前请确保机器0和机器1在同一网段且网络互通，其他参数设置请保持一致。
启动时先启动机器0上模型运行命令，然后启动机器1上模型运行命令（机器0会等待机器1的进程加入）。

## 精度对比模式说明

1.npu侧，修改/code/overall/LLM_deepspeed_npu.yaml中参数配置。

```shell
precision_mode_enabled: True  # 默认为False，精度对其模式下修改为True。
```

2.gpu侧，下载官方模型代码后，并使用hllm_gpu_precision.patch文件进行修改。

```shell
git clone https://github.com/bytedance/HLLM.git
cd HLLM && git checkout 864f1722
cp -f ../hllm_gpu_precision.patch ./ && git apply hllm_gpu_precision.patch
```

3.参数对齐

（1）确保数据集和预训练模型一致，如均使用amazon_books数据集和TinyLlama1B预训练模型。

（2）确保训练参数一致，对比gpu侧/code/overall/LLM_deepspeed.yaml与npu侧/code/overall/LLM_deepspeed_npu.yaml中同名参数对应一致；/code/HLLM/HLLM.yaml中对应参数一致。

4.运行模型

npu侧按以上介绍，启动模型;gpu侧按开源介绍启动模型训练即可。
