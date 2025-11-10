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
    "data_type": "fp32",
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
+ type：推理还是训练模式
+ epoch: 训练步数或者推理循环次数
+ profiling_flag: 是否抓取profiling
+ compile_flag: 是否需要使能图编译
+ aclgraph_flag: 是否需要使能图下沉
+ data_type: 模型的input数据类型
+ run_cmd: 模型的运行命令
+ pip_install_self: 是否依赖安装开源仓自己
+ pip_install_requirements: 开源仓是否要安装起根目录下的requirements.txt依赖包
+ extra_cmd: 适配npu需要额外执行的命令

