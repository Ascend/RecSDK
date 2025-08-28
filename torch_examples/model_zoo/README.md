# MMOE模型和ETA模型支持
本文档主要介绍如何进行推荐系统模型中的mmoe模型和eta模型的数据预处理和模型训练。


## 版本配套说明
本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件,源码编译需使用指定版本的Python、GCC、CMake等工具,仅支持昇腾平台（Atlas 800T A2）,基于软件环境以RecSDK-Torch提供的基础镜像环境为准，主要的配套依赖如下表所示：

| Python版本   | 主要配套依赖                                |
|------------|---------------------------------------|
| Python3.11 | torch==2.6.0<br/>torch_npu==2.6.0<br/> |

### 基础镜像
下载基础镜像地址为：https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5

### 启动容器
说明：以下启动命令仅作参考
```shell
#!/bin/bash
container_name=$1
image_name=$2
docker run \
-it \
--name "${container_name}" \
-e ASCEND_VISIBLE_DEVICES=0-7 \
--shm-size="300g" \
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```
执行如下命令新建容器：
```shell
source /etc/profile
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

### 设置环境变量
进入容器后，设置环境变量
```shell

source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

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

