# torchrec适配npu说明

## 代码结构

```shell
.
├── build_whl_torchrec1.6.0.sh   # torchrec==1.6.0+npu构建脚本
├── torchrec1.6.0_npu.patch      # 基于torchrec1.6.0,适配NPU修改后的patch文件
└── README.md                    # 适配说明
```

## 说明

1.本样例提供了一个简洁的torchrec适配到昇腾设备方案，用户可快速编译、安装torchrec的npu适配版本。

2.本样例基于开源torchrec v1.6.0版本进行适配。配套开源软件版本：torch 2.11.0, fbgemm_gpu 1.6.0+cpu。

3.本样例依赖fbgemm算子库，使用前请参考[安装教程](https://gitcode.com/Ascend/fbgemm-ascend)完成算子安装。

## 编译安装

### 基于torchrec v1.6.0版本

```shell
# 在当前目录下载源码并编译
git clone https://github.com/pytorch/torchrec.git
cd torchrec && git checkout 16fd38e5 && cd ..
dos2unix build_whl_torchrec1.6.0.sh && bash build_whl_torchrec1.6.0.sh
# 安装
cd torchrec/dist
pip3 install torchrec-1.6.0+npu-*.whl
# 安装torchrec依赖。若提示安装expecttest失败，可忽略，该包仅测试场景使用
pip3 install -r requirements.txt
```
