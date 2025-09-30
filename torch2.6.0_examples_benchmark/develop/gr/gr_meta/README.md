# GR模型迁移样例说明

## 适配说明

本样例的适配对象为Meta开源的Generative Recommendations模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。
模型参考的开源链接为 https://github.com/facebookresearch/generative-recommenders
克隆源码并固定版本为:Commits on Dec 16, 2024，提交的SHA-1 hash值（提交ID）：bb389f9539b054e7268528efcd35457a6ad52439


## 代码结构说明

```shell
├── gr_npu.patch    # 模型迁移适配patch文件
├── README.md       # 样例迁移说明文档
└── run.sh          # 模型运行脚本
```

## 运行环境准备
请参考：[模型样例运行环境说明](../README.md)

## GR源码适配

将 Generative Recommendations 模型迁移到NPU上并适配NPU融合HSTU算子，代码修改部分已经编写在`gr_npu.patch`中，载入命令如下：

```bash
# 进入当前目录，克隆meta开源GR模型源代码
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders && git checkout bb389f9539b054e7268528efcd35457a6ad52439
cp ../gr_npu.patch ./ && git apply gr_npu.patch
```

## 安装模型依赖python包
```bash
pip3 install -r requirements.txt
```
说明:本模型样例是迁移NPU适配，并在pytorch框架下2.6.0配套版本运行，已忽略nvidia和tensorflow相关安装包,并调整部分配套依赖包版本。
## 数据集准备
参考源码，在preprocess_public_data.py同级目录下执行如下命令。
```shell
mkdir -p tmp/ && python3 preprocess_public_data.py
```
说明：本次测试基于ml-1m数据集，整体使用hstu-sampled-softmax-n128-large-final.gin参数配置，为适配npu算子调整个别参数见patch文件。

## 模型运行

```shell
# 拷贝运行脚本到当前目录
cp ../run.sh ./
```
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
```shell
bash run.sh
```

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
