# compute_divide 使用说明

本文档说明如何在算力切分环境下完成 DLRM 的训练与推理。

## 前置条件

1. 已按 [DLRM 运行说明](../README.md) 成功跑通 DLRM 模型。
2. 已按 [算力切分环境配置](https://www.hiascend.com/developer/techArticles/20251212-1?sectionId=0101178462695499013&classId=) 完成算力切分环境配置。

## DLRM 源码适配

在 `compute_divide` 同级目录下执行：克隆官方 DLRM 仓库、切到指定提交，并应用算力切分 patch。

```shell
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99
cp -f ../dlrm_divide.patch ./
git apply dlrm_divide.patch
```

## 训练流程

### 步骤 1：拷贝 run.sh 到 torchrec_dlrm

将本目录下的 `run.sh` 拷贝到已适配好的 `torchrec_dlrm` 目录（与 `dlrm_main.py` 同级），后续在该目录下按需修改并运行。

```bash
cp -f run.sh <target>/dlrm/dlrm/torchrec_dlrm/
```

### 步骤 2：配置 run.sh 进行训练

在 `run.sh` 中设置训练相关变量（若已有训练好的 checkpoint，可跳过训练，直接进行推理）：

```bash
export PREPROCESSED_DATASET=<数据集路径>
export EXECUTE_TRAIN=1
export EXECUTE_SING_CARD_EVAL=0
export WITH_EMBCACHE_AND_LOAD=0
export WITH_EMBCACHE_AND_SAVE=1
# Embcache 设备端内存大小（字节），例如 8589934592（8GB）
# 以下变量请根据实际环境修改
export EMBCACHE_SIZE_ON_DEVICE_MEM=17179869184
export WORLD_SIZE=4
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

### 步骤 3：进入训练容器

将 `xxxxx` 替换为实际训练容器名称。

```bash
docker exec -it xxxxx /bin/bash
```

### 步骤 4：（可选）在容器内设置环境变量

若遇库或命令找不到，可在容器内执行：

```bash
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.11.0/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 步骤 5：在容器内运行训练脚本

将 `<torchrec_dlrm_dir>` 替换为 torchrec_dlrm 的实际路径。

```bash
cd <torchrec_dlrm_dir>
bash run.sh
```

## 推理流程（算力切分）

### 步骤 6：进入算力切分推理容器

将 `xxxxx` 替换为实际推理容器名称。

```bash
docker exec -it xxxxx /bin/bash
```

### 步骤 7：配置 run.sh 进行推理

在 `run.sh` 中改为推理模式，并设置为单卡：

```bash
export PREPROCESSED_DATASET=<数据集路径>
export EXECUTE_TRAIN=0
export EXECUTE_SING_CARD_EVAL=1
export WITH_EMBCACHE_AND_LOAD=1
export WITH_EMBCACHE_AND_SAVE=0
# Embcache 设备端内存大小（字节），例如 8589934592（8GB）
# 推理仅支持单卡
export EMBCACHE_SIZE_ON_DEVICE_MEM=8589934592
export WORLD_SIZE=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

随后在容器内进入 torchrec_dlrm 目录并执行 `bash run.sh`。

## 预期结果

训练或推理结束后，终端会打印 AUROC，数值在预期范围内即表示流程正常。

## 声明

受算力切分功能限制，本样例仅用于验证算力切分场景下的精度与流程，请勿随意修改运行配置或用于非验证用途。
