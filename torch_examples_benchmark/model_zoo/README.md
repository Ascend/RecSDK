# MMOE模型和ETA模型
本文档主要介绍如何进行推荐系统模型中的mmoe模型和eta模型的数据预处理和模型训练

## 主要依赖
**Pytorch:** 2.6.0
### 安装pytorch
```commandline
pip3 install torch==2.6.0+cpu  --index-url https://download.pytorch.org/whl/cpu # x86
pip3 install torch==2.6.0 # arm
```
### 安装必要依赖
```commandline
pip3 install -r requirements.txt
```

## 数据集准备
1. 准备alicpp数据集
2. 对于[Ali-CPP](https://tianchi.aliyun.com/dataset/408)数据集，我们提供完整的预处理流程[参考](https://gitee.com/ascend/RecSDK/blob/develop/examples/rec_model_zoo/behaviour_and_multi_task/data/aliccp/README.md)，
下载数据集至alicpp目录,如：
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
3.进入aliccp目录执行如下命令：
```commandline
bash run.sh 
```
执行完成后预处理后的数据集会生成到指定目录，本用例默认生成在aliccp_out目录。
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

