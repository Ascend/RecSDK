# MMOE模型和ETA模型支持
本文档主要介绍如何进行推荐系统模型中的mmoe模型和eta模型的数据预处理和模型训练。

## 运行环境准备
请参考：https://gitcode.com/Ascend/RecSDK/tree/develop/torch_examples/README.md

说明:如果只执行mmoe和eta模型样例，可以忽略torchrec,hybrid_torchrec以及算子等依赖的安装。

## 模型运行

进入模型适配目录
### 安装必要依赖
```commandline
pip3 install -r requirements.txt
```

### 数据集准备
1. MMOE模型和ETA模型均以公开的点击与转化预估数据集作为基础数据集。
2. 对于[Ali-CPP](https://tianchi.aliyun.com/dataset/408)数据集，我们提供完整的预处理流程。
1.下载以上链接数据集sample_test.tar.gz和sample_train.tar.gz至alicpp目录，解压后如下结构。：
```commandline

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
2.进入aliccp目录执行如下命令，做数据预处理。
```commandline
bash run.sh 
```
执行完成后预处理后的数据集会生成到指定目录，本用例默认生成在aliccp_out目录下。

## 模型训练
1.执行训练脚本，传入模型所需参数，参考命令如下：
```commandline
# mmoe模型
python3 mmoe.py --data_dir aliccp/aliccp_out/ --train_batch_num 2000 --eval_batch_num 20  # 根据实际情况传入参数

# eta模型
python3 eta.py --data_dir aliccp/aliccp_out/ --train_batch_num 2000 # 根据实际情况传入参数
```
2.参数说明
```commandline
# 通过以下命令方式查看参数及默认值情况
python3 mmoe.py  --help 
```
