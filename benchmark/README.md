# 代码结构

```shell
|-- benchmark/
    |-- ckpt/        # 权重文件目录
    |-- configs/     # 框架相关配置文件目录
    |-- data/        # 数据集目录
    |-- models/      # 存放模型代码、模型规格配置文件的目录
    |-- patches/     # 模型迁移适配patch文件目录
    |-- test/        # 模型测试脚本目录
    |-- tools/       # 工具类脚本目录
    |-- README.md    # 模型迁移说明文档
    |-- run.py       # 模型运行脚本
```

# 运行环境准备

参考[RecSDK-Torch 模型样例运行环境说明](../torch2.6.0_examples_benchmark/develop/README.md)

# 必要依赖安装

```shell
apt-get install protobuf-compiler protobuf-devel
pip install tensorflow
pip install pytest
```

# quick start

xxx.json替換为为configs目录下的配置文件名；

```shell
python run.py xxx.json --eager
```

**xxx.json**:为configs文件夹中的配置文件
**--eager**：为强制跑eager模式，不配置默认跑inductor模式（当前未支持）
**--no_hf32**：禁用混合精度加速，不配置默认使能混合精度加速。NCF、DIN-pytorch、Multitask-Recommendation-Library比对GPU与NPU精度时需禁用
**--custom_dropout**：使用自定义的dropout函数，不配置使用默认的dropout函数。Multitask-Recommendation-Library比对GPU与NPU精度是需配置

# 性能指标

模型正常运行后，会在models目录下生成性能相关文件，目录为./models/save_results_{device_name}/performance_result.txt,其中{device_name}为运行设备名称，如npu、cuda、cpu,文件内容为模型的性能指标，如推理时间、qps等。
如config中开启了profiling_flag，会在./models/profiling/{model_name}生成profiling结果文件。其中{model_name}为模型名字。

# 精度指标

模型正常运行后，会在models目录下生成落盘输出文件，目录为save_results_{device_name}/{model_name},其中device_name为运行设备名称，如npu、cuda、cpu,{model_name}为模型名字。

使用tools目录下的脚本对两份数据进行对比，cpu的数据作为标杆，对比npu数据与cpu数据的差异。
对比脚本为tools/accuracy_compare.py，使用方法为：

```shell
python accuracy_compare.py --actual_output ./models/save_results_npu --expected_output ./models/save_results_cpu  --rtol 1e-4 --atol 1e-4
```

其中--actual_output为npu数据目录，--expected_output为cpu数据目录，--rtol为相对误差容忍度，--atol为绝对误差容忍度。

# config文件示例

```json
{
    "name": "AutoInt",
    "url": "https://github.com/reczoo/FuxiCTR.git",
    "commit_id": "b7dff736885fdb8f59387d82d08219ad2e4cae50",
    "patch_path": "patches/fuxictr_npu.patch",
    "type": "infer",
    "epoch": 100,
    "profiling_flag": true,
    "aclgraph_flag": true,
    "data_type": "float32",
    "run_cmd": [
        "python",
        "model_zoo/DMR/run_expid.py",
        "--expid",
        "DMR_test",
        "--gpu",
        "0"
    ],
    "pip_install_self": true,
    "pip_install_requirements": true,
    "extra_cmd": ["pip install Scikit-learn<1.5"]
}
```

+ name:模型名字
+ url:模型代码仓下载路径
+ commit_id:本示例适配的commit节点
+ patch_path:适配的patch目录
+ type：推理还是训练模式(infer/train/train_evaluate)
+ epoch: 训练步数或者推理循环次数
+ profiling_flag: 是否抓取profiling
+ aclgraph_flag: 是否需要使能图下沉
+ data_type: 模型的input数据类型(float32/float16/bfloat16)
+ run_cmd: 模型的运行命令
+ pip_install_self: 是否依赖安装开源仓自己
+ pip_install_requirements: 开源仓是否要安装起根目录下的requirements.txt依赖包
+ extra_cmd: 适配npu需要额外执行的命令

# 模型额外操作说明

## yolov5 模型

yolov5模型需手动下载权重文件。https://gitcode.com/open-source-toolkit/6e474/blob/main/yolov5%20%E5%AE%98%E6%96%B9%E6%9D%83%E9%87%8D%E6%96%87%E4%BB%B6.zip
从链接里下载并解压，把yolov5s.pt放在ckpt文件夹下再运行

## Multitask-Recommendation-Library(MMOE) 模型

MMOE模型运行前需访问 https://tianchi.aliyun.com/dataset/74690 下载aliexpress_NL_datasets.zip数据集，
并将数据集放到**与本README.md同级的data目录**下。

## DLRM_META 模型

DLRM_META模型运行前需参考 https://github.com/facebookresearch/dlrm.git 准备Kaggle Display Advertising dataset数据集(https://ailab.criteo.com/ressources/), 
在**与本README.md同级的data目录**下，新建dlrm_meta目录存放原始train.txt和test.txt文件。

## RECSYS_RANKING 模型

RECSYS_RANKING模型运行前需参考 https://github.com/NVIDIA/recsys-examples/blob/v25.09/examples/hstu/README.md#dataset-preprocessing 准备数据集, 
在**与本README.md同级的data目录**下，新建recsys_ranking目录存放。
