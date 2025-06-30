# NPU融合算子适配GR

## 适配说明

本样例的适配对象为Generative Recommendations模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。

模型参考的开源链接为 https://github.com/facebookresearch/generative-recommenders

克隆源码并固定版本为:Commits on Dec 16, 2024，提交的SHA-1 hash值（提交ID）：bb389f9539b054e7268528efcd35457a6ad52439

验证运行的算力平台：Atlas A2训练系列产品

## 启动容器

镜像下载地址： https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5

该镜像中部分配套版本说明：

| 软件包简称   | 配套版本   |
|---------|--------|
| Pytorch | 2.1.0  |
| Python  | 3.11.0 |
| Fbgemm  | 0.5.0  |

启动容器命令参考：

```python
docker run \
-u root \
-it \
--name ${container_name} \
--net=host \
--shm-size="300g" \
--privileged \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/ascend:/root/ascend \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
```

## 安装依赖

### CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| CANN-toolkit | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.0.beta1&driver=1.0.28.alpha |

请根据机器架构、机器型号在下载链接中选择合适的安装包进行安装。

### 安装torch_npu

下载最新版本的mindxsdk-mxec-add-ons安装包

解压之后，进入文件夹mindxsdk-mxec-add-ons：

`cd mindxsdk-mxec-add-ons`

`cd torch_plugin`

执行命令安装：

`pip3 install torch_npu-2.1.0.post9-cp311-cp311-linux_aarch64.whl`

### 安装算子

重新进入文件夹 mindxsdk-mxec-add-ons, 安装需要的昇腾适配算子： jagged_to_padded_dense、IndexSelect优化、
dense_to_jagged、asynchronous_complete_cumsum、gather_for_rank1

```shell
cd mindxsdk-mxec-add-ons/mxrec_ops
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_gather_for_rank1.run
bash mxrec_opp_hstu_dense_forward.run
bash mxrec_opp_hstu_dense_backward.run
```

### 编译融合算子依赖的lib

进入 torch_library 文件夹：

```shell
cd mindxsdk-mxec-add-ons/torch_library
cd 2.1.0/hstu
bash build_ops.sh
```

执行完以上命令之后，融合算子的依赖包libhstu_dense_ops.so会生成在同目录下的build文件夹下，可将该so包拷贝到某固定目录下。示例如下：

`cp ./build/libhstu_dense_ops.so /home/torch_ops/`

## 代码修改

将 Generative Recommendations 模型迁移到NPU上并适配NPU融合HSTU算子，代码修改部分已经编写在`NPU_GR.patch`中，载入命令如下：

```bash
git checkout bb389f9539b054e7268528efcd35457a6ad52439
git apply NPU_GR.patch
```

## 测试示例

本次测试基于ml-1m数据集，使用NPU的HSTU融合算子(去rab, 下三角mask)， 基于以下配置config文件进行测试：

### config文件：

创建一个hstu-mt-3400.gin文件，文件内容如下。将该gin文件放置在 `generative-recommenders/configs/ml-1m/` 目录下

```gin
train_fn.dataset_name = "ml-1m"
train_fn.max_sequence_length = 3389
train_fn.local_batch_size = 32

train_fn.main_module = "HSTU"
train_fn.dropout_rate = 0.2
train_fn.user_embedding_norm = "l2_norm"
train_fn.num_epochs = 1
train_fn.item_embedding_dim = 512

hstu_encoder.num_blocks = 3
hstu_encoder.num_heads = 2
hstu_encoder.dqk = 256
hstu_encoder.dv = 256
hstu_encoder.linear_dropout_rate = 0.2

train_fn.learning_rate = 1e-3
train_fn.weight_decay = 0
train_fn.num_warmup_steps = 0

train_fn.interaction_module_type = "DotProduct"
train_fn.top_k_method = "MIPSBruteForceTopK"

train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 128
train_fn.eval_interval = 50
train_fn.sampling_strategy = "local"
train_fn.temperature = 0.05
train_fn.item_l2_norm = True
train_fn.l2_norm_eps = 1e-6

train_fn.enable_tf32 = True
train_fn.precision_mode = False

create_data_loader.prefetch_factor = 128
create_data_loader.num_workers = 8
```

### 运行命令：

修改run.sh 脚本，使用上面的配置文件：

```shell
export USE_NPU_HSTU=1
export ENABLE_RAB=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 main.py --gin_config_file=configs/ml-1m/hstu-mt-3400.gin --master_port=12345 | tee temp.log
```

执行命令：
`bash run.sh`

### 性能测试结果

| 数据集   | seq_len | num_block | num_heads | dqk、dv | 端到端耗时  | GPU triton耗时 |
|-------|---------|-----------|-----------|--------|--------|--------------|
| ml-1m | 3400    | 3         | 2         | 256    | 54.8ms | 75ms         |

### 精度loss比对

精度对比需额外适配，详见：https://gitee.com/ascend/RecSDK/blob/branch_v7.0.0-POC_torch/torch/examples/generative_recommenders/gpu/README.MD#precision_mode%E6%A8%A1%E5%BC%8F

| Step | GPU loss | NPU loss | Loss diff              |
|------|----------|----------|------------------------|
| 0    | 4.859579 | 4.859569 | 1.0000000000509601e-05 |
| 10   | 4.85863  | 4.858632 | 2.000000000279556e-06  |
| 20   | 4.855053 | 4.855056 | 3.000000000419334e-06  |
| 30   | 4.850527 | 4.850548 | 2.1000000000270802e-05 |
| 40   | 4.837311 | 4.837361 | 4.999999999988347e-05  |
| 50   | 4.821918 | 4.822002 | 8.400000000019503e-05  |
| 60   | 4.802688 | 4.80268  | 8.000000000230045e-06  |
| 70   | 4.776829 | 4.777043 | 0.00021399999999971442 |
| 80   | 4.760865 | 4.760607 | 0.0002579999999996474  |
| 90   | 4.713118 | 4.712983 | 0.0001349999999993301  |
| 100  | 4.680532 | 4.68044  | 9.200000000042508e-05  |

## FAQ
