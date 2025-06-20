# Dcnv2模型迁移样例
## 迁移说明
本样例以DLRM模型为例,适配hybrid_torchrec框架并在NPU上进行训练。  
模型参考的开源链接为:https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/,  
克隆源码并固定版本为:Commits on Jun 7 , 2024，提交的SHA-1 hash值（提交ID）：b631a99 

## 前提条件
需要用户构建容器基础环境，参考RecSDK/torchrec/docker/README.md

## 数据集下载
### 使用官网数据集
进入[开源模型官网](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD)，官网提供两种方式跑通demo：
1. 下载原始数据处理后，提前进行mutil-hot的合成，产生4T的数据
2. 下载原始数据处理后，在训练的过程中生成mutil-hot数据，使用690gb数据集
由于1需要的条件苛刻，大部分机器很难满足条件，本次演示使用2中的条件。无host瓶颈的情况下，对性能影响较小。需要修改模型脚本代码，让host生成的数据在pin_memory上。
### 使用随机数据集
如果用户仅验证功能，可以使用生成的随机数据集。
```shell
mkdir generate_data
cp generate_data.py generate_data
cd generate_data
python3 generate_data.py
```

## 修改脚本并运行
下载官方模型代码，并使用patch进行修改
```shell
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99 
cp -f ../dcnv2_recsdk_torch.patch ./
git apply dcnv2_recsdk_torch.patch
```

在dlrm/torchrec_dlrm下运行如下脚本启动训练任务。`$insert_your_path_here`为数据集路径。

```shell
export PREPROCESSED_DATASET=$insert_your_path_here
export TOTAL_TRAINING_SAMPLES=4195197692 ;
export GLOBAL_BATCH_SIZE=16384;
export WORLD_SIZE=8;
torchx run -s local_cwd dist.ddp -j 1x${WORLD_SIZE} --script dlrm_main.py -- \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --in_memory_binary_criteo_path $PREPROCESSED_DATASET \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005 \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 2>&1 | tee "temp.log"
```
