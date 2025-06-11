# GR模型NPU适配

## 适配说明

本样例的适配对象为Generative Recommendations模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。

模型参考的开源链接为 https://github.com/facebookresearch/generative-recommenders

克隆源码并固定版本为:Commits on Dec 16, 2024，提交的SHA-1 hash值（提交ID）：bb389f9539b054e7268528efcd35457a6ad52439

验证运行的算力平台：Atlas A2训练系列产品

## 版本配套说明

### 软件配套说明

| 软件包简称   | 配套版本   |
|---------|--------|
| Python  | 3.11.0 |
| Pytorch | 2.6.0  |
| Fbgemm  | 1.1.0  |

### CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| CANN-toolkit | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.0.beta1&driver=1.0.28.alpha |

## 安装依赖

请根据机器架构、机器型号选择合适的安装包进行安装。

### 安装Pytorch配套

```shell
# torch版本
pip3 install torch==2.6.0+cpu  --index-url https://download.pytorch.org/whl/cpu # x86
pip3 install torch==2.6.0 # arm
# fbgemm_gpu版本
pip3 install fbgemm_gpu==1.1.0+cpu -i https://download.pytorch.org/whl/cpu
# torch_npu
pip3 install torch_npu-2.6.0.*.whl
```
### 安装算子

进入mindxsdk-mxec-add-ons/mxrec_ops文件夹, 安装需要的昇腾适配算子： jagged_to_padded_dense、IndexSelect优化、
dense_to_jagged、asynchronous_complete_cumsum、gather_for_rank1、hstu前向和反向算子。

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

### 编译算子适配层文件

进入mindxsdk-mxec-add-ons/torch_plugin/torch_library/2.6.0/common文件夹,编译适配层文件。

```shell
cd mindxsdk-mxec-add-ons/torch_plugin/torch_library/2.6.0/common
bash build_ops.sh
```

执行完以上命令之后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，以及python默认安装的site-package路径。也可将该so包拷贝到某固定目录下便于在代码中加载。示例如下：
```shell
torch.ops.load_library(/path/to/libfbgemm_npu_api.so) # 根据实际路径
```

## GR源码适配

将 Generative Recommendations 模型迁移到NPU上并适配NPU融合HSTU算子，代码修改部分已经编写在`gr_npu.patch`中，载入命令如下：

```bash
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders && git checkout bb389f9539b054e7268528efcd35457a6ad52439
cp ../gr_npu.patch ./ && git apply gr_npu.patch
```

## 安装模型依赖python包
```bash
pip3 install --ignore-install -r requirements.txt
```
说明:本模型样例是迁移NPU适配，并在pytorch框架下运行，可手动注释忽略nvidia和tensorflow相关安装包,torch使用2.6.0版本配套。
## 数据集准备
参考源码，在preprocess_public_data.py同级目录下执行如下命令。
```shell
mkdir -p tmp/ && python3 preprocess_public_data.py
```
说明：本次测试基于ml-1m数据集，整体使用hstu-sampled-softmax-n128-large-final.gin参数配置，为适配npu算子调整个别参数见patch文件。

## 模型运行

修改run.sh 脚本：

```shell
export USE_NPU_HSTU=1                                                                 # 是否使用hstu算子加速
export ENABLE_RAB=0                                                                   # 是否带RAB
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"                     # 根据实际情况修改
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3                                              # 根据实际情况修改
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345 | tee temp.log
```

拷贝run.sh与main.py同级目录，执行命令：
`bash run.sh`

### 整网精度
MovieLens-1M (ML-1M):

| Method   | NDCG@10 | NDCG@50 | HR@10  | HR@50  | MRR    |
|-------|---------|---------|--------|--------|--------|
|HSTU-large| 0.1531  | 0.2142  | 0.2772 | 0.5531 | 0.1312 |
说明:以上为hstu-sampled-softmax-n128-large-final.gin参数配置,训练一轮数据的测试精度。


### 性能参考
| Steps | NPU适配 | HSTU算子加速 |
|-------|-------|----------|
| 100   | 68.59 | 16.96    | 

说明：以上表示每100步耗时，单位：秒。
## FAQ
