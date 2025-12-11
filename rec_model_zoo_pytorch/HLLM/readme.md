# HLLM模型说明

## 主要依赖
**Python:** 3.12
**Pytorch:** 2.7


## 原代码
https://github.com/bytedance/HLLM.git

## patch 使用
进入HLLM原项目目录
 ```shell
 git checkout 3f4fc3e5
 git apply HLLM.patch
 ```

## 运行
使用code目录下的运行脚本launch_gpu.sh或launch_npu.sh开始推理
 ```shell
 bash launch_npu.sh
 ```

## 环境准备
运行前，需要先在环境变量设置设备类型(launch脚本中已配置)

GPU:
 ```shell
 export DEVICE_TYPE="cuda"
 ```

NPU:
 ```shell
 export DEVICE_TYPE="npu"
 ```

## 关键参数介绍
| 参数名称                 | 参数说明               |
|----------------------|--------------------|
| device_type          | 设备类型，cuda或npu      |
| config_file          | 配置文件               |
| MAX_ITEM_LIST_LENGTH | 物品列表最大长度           |
| MAX_TEXT_LENGTH      | 最大文本长度             |
| checkpoint_dir       | 权重保存目录             |
| item_pretrain_dir    | 物品预训练目录            |
| user_pretrain_dir    | 用户预训练目录            |
| text_path            | text目录             |
| data_path            | 数据集目录              |
| val_only             | 训练模式               |
| enable_cudagraph     | 启用cudagraph(gpu专属) |
| enable_aclgraph      | aclgraph(npu专属)    |
| enable_profiler      | 启用性能采集             |
| profiler_batch_count | 性能采集batch数         |
| eval_batch_size      | 推理batch size       |
