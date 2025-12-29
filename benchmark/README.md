# 代码结构
```shell
|-- benchmark/
    |-- configs/     # 框架相关配置文件目录
    |-- models/      # 存放模型代码、模型规格配置文件的目录
    |-- patches/     # 模型迁移适配patch文件目录
    |-- README.md    # 模型迁移说明文档
    |-- run.py       # 模型运行脚本
```

# 运行环境准备
参考[RecSDK-Torch 模型样例运行环境说明](../torch2.6.0_examples_benchmark/develop/README.md)

# 必要依赖安装

```shell
apt-get install protobuf-compiler
pip install tensorflow
pip install pytest
```


# quick start
xxx.json替換为为configs目录下的配置文件名；
```shell
python run.py  xxx.json
```

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
    "compile_flag": true,
    "aclgraph_flag": false,
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
+ compile_flag: 是否需要使能图编译
+ aclgraph_flag: 是否需要使能图下沉
+ data_type: 模型的input数据类型(float32/float16/bfloat16)
+ run_cmd: 模型的运行命令
+ pip_install_self: 是否依赖安装开源仓自己
+ pip_install_requirements: 开源仓是否要安装起根目录下的requirements.txt依赖包
+ extra_cmd: 适配npu需要额外执行的命令

## yolov5 模型需要手动下载权重文件

https://gitcode.com/open-source-toolkit/6e474/blob/main/yolov5%20%E5%AE%98%E6%96%B9%E6%9D%83%E9%87%8D%E6%96%87%E4%BB%B6.zip
从链接里下载并解压，把yolov5s.pt放在ckpt文件夹下再运行

## SIM 模型gpu运行需要应用patch
```
cd ./benchmark/models/SIM/
cp ../../patches/sim_gpu.patch ./ && git apply sim_gpu.patch
```