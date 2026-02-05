# torchrec适配npu说明

## 代码结构
```shell
.
├── build_whl_torchrec1.1.0.sh   # torchrec==1.1.0+npu构建脚本
├── build_whl_torchrec1.2.0.sh   # torchrec==1.2.0+npu构建脚本
├── torchrec1.1.0_npu.patch      # 基于torchrec1.1.0,适配NPU修改后的patch文件
├── torchrec1.2.0_npu.patch      # 基于torchrec1.2.0,适配NPU修改后的patch文件
└── README.md                    # 适配说明
```

## 说明
1.本样例提供了一个简洁的torchrec适配到昇腾设备方案，用户可快速编译、安装torchrec的npu适配版本。

2.本样例基于开源torchrec项目多个版本进行了适配，用户可根据需要自行选择配套版本。

2.1 基于torchrec v1.1.0版本进行适配。配套开源软件版本：torch 2.6.0, fbgemm_gpu 1.1.0+cpu。

2.2 基于torchrec v1.2.0版本进行适配。配套开源软件版本：torch 2.7.1, fbgemm_gpu 1.2.0+cpu。

## 编译安装
### 基于torchrec v1.1.0版本
```shell
# 在当前目录下载源码并编译
git clone -b release/v1.1.0 https://github.com/pytorch/torchrec.git
cd torchrec && git checkout 2c5f6ee && cd ..
bash build_whl_torchrec1.1.0.sh
# 安装
cd torchrec/dist
pip3 install torchrec-1.1.0+npu-*.whl
# 安装torchrec依赖。若提示安装expecttest失败，可忽略，该包仅测试场景使用
pip3 install -r requirements.txt 
```

### 基于torchrec v1.2.0版本
```shell
# 在当前目录下载源码并编译
git clone -b release/v1.2.0 https://github.com/pytorch/torchrec.git
cd torchrec && git checkout 5db1a21 && cd ..
bash build_whl_torchrec1.2.0.sh
# 安装
cd torchrec/dist
pip3 install torchrec-1.2.0+npu-*.whl
# 安装torchrec依赖。若提示安装expecttest失败，可忽略，该包仅测试场景使用
pip3 install -r requirements.txt
```
