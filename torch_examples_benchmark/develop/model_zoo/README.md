# MMOE模型和ETA模型支持

本文档主要介绍如何进行推荐系统模型中的mmoe模型和eta模型的数据预处理和模型训练。

## 运行环境准备

参考[README](../../develop/README.md)

说明:如果只执行mmoe和eta模型样例，可以忽略torchrec,hybrid_torchrec以及算子等依赖的安装。

## 模型运行

进入模型适配目录

### 安装必要依赖

```bash
pip3 install -r requirements.txt
```

### 数据集准备

1. MMOE模型和ETA模型均以公开的点击与转化预估数据集作为基础数据集。
2. 对于[Ali-CPP](https://tianchi.aliyun.com/dataset/408)数据集，我们提供完整的预处理流程。
   
    下载以上链接数据集sample_test.tar.gz和sample_train.tar.gz（2个压缩包共约8.9GB）至当前路径下的aliccp目录。
    
    进入aliccp目录，并对两个压缩包解压，指令示例：`tar -zxvf sample_test.tar.gz`和`tar -zxvf sample_train.tar.gz`，解压后文件大小约161GB，文件结构如下：

    ```text
    ├── common_features_test.csv
    ├── common_features_train.csv
    ├── sample_skeleton_test.csv
    ├── sample_skeleton_train.csv
    ├── step1_count_vocabs.py
    ├── step2_remove_low_ids.py
    ├── step3_map_ids.py
    ├── step4_split_val.py
    ├── step5_merge_table.py
    ├── step6_gen_torch_dataset.py
    ├── step7_gen_spec.py
    └── run.sh
    ```

    执行如下命令，做数据预处理。

    ```bash
    bash run.sh
    ```

    执行后预处理的数据集会生成到指定目录，本用例默认生成在aliccp/aliccp_out目录下，处理后的数据集大小约177GB，预处理耗时约2.5-3h。

## 模型训练

执行训练脚本，传入模型所需参数，参考命令如下：

```bash
# mmoe模型
python3 mmoe.py --data_dir aliccp/aliccp_out/ --train_batch_num 2000 --eval_batch_num 20  # 根据实际情况传入参数

# eta模型
python3 eta.py --data_dir aliccp/aliccp_out/ --train_batch_num 2000 # 根据实际情况传入参数
```

参数说明

```bash
# 通过以下命令方式查看参数及默认值情况
python3 mmoe.py  --help 
```
