# torchrec适配npu说明

## 代码结构
```shell
.
├── build_whl.sh            # torchrec==1.1.0+npu构建脚本
├── torchrec_npu.patch      # 侵入式修改patch文件
└── README.md               # 适配说明
```
## 编译安装
```shell
# 在当前路径下载固定分支torchrec源码并编译
git clone -b release/v1.1.0 https://github.com/pytorch/torchrec.git
cd torchrec && git checkout 2c5f6ee && cd ..
bash build_whl.sh
# 若已安装torchrec，需先卸载
pip3 uninstall torchrec -y
# 安装
cd torchrec/dist
pip3 install torchrec-1.1.0+npu-*.whl
# 安装torchrec依赖。若提示安装expecttest失败，可忽略，该包仅测试场景使用
pip3 install -r requirements.txt 
```

若环境无网络，采用“手动下载torchrec源码并上传到目标环境”的方式获取源码时，需注意：
> 1. 下载源码包时需切换分支后再下载（patch文件是基于指定分支修改，不适配torchrec仓库默认的master分支）。
> 2. 在目标环境上解压缩后，需将解压出来的文件夹名称改为`torchrec`，使文件夹名称和上述指令及build_whl.sh脚本中的文件夹名称一致。

## 说明
1.本样例提供了一个简洁的torchrec适配到昇腾设备方案，用户可快速编译、安装torchrec的npu适配版本。

2.本样例是基于开源torchrec项目v1.1.0版本基础上进行的适配。配套使用的torch版本为2.6.0,fbgemm_gpu版本为1.1.0。
