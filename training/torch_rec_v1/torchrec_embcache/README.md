# TorchRec-EmbCache NPU适配方案

## 软件介绍
本项目是基于开源项目TorchRec的1.1.0/1.2.0版本开发的embedding多级缓存扩展,（参考网站：https://github.com/pytorch/torchrec ）为助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理,
Python版本要求：Python >= 3.11。

基于PyTorch开源软件版本，支持两种软件版本配套，可根据需要自行选择。

| 配套版本  | PyTorch | torch-npu | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本1 | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           | 1.1.0             |
| 配套版本2 | 2.7.1   | 2.7.1     | 1.2.0+npu | 1.2.0      | 1.2.0           | 1.2.0             |

### 1.环境准备

参考 [Rec SDK文档](https://www.hiascend.com/document/detail/zh/mindsdk/71rc1/rec/recug/mxrectorch_0014.html) 的"制作Rec SDK Torch训练镜像"和"启动容器"章节。

### 2.软件包安装
参考 [Rec SDK文档](https://www.hiascend.com/document/detail/zh/mindsdk/71rc1/rec/recug/mxrectorch_0014.html) 的“安装Rec SDK Torch”章节。

#### 2.1 前提条件
torchrec_embcache依赖于hybrid_torchrec包，请先参考[hybrid_torchrec README](../hybrid_torchrec/README.MD)完成hybrid_torchrec及其依赖包安装。

本章节仅介绍torchrec_embcache软件包安装。

#### 2.2 torchrec_embcache安装
- 基于软件包安装

从[RecSDK release版本](https://gitcode.com/Ascend/RecSDK/releases)，下载hybrid_torchrec软件包tar.gz包。

tar.gz压缩包解压后包含torchrec_embcache*.whl包。

```shell
# 如果已安装，请先卸载
pip3 uninstall -y torchrec_embcache
tar zxvf Ascend-mindxsdk-hybrid-torchrec*.tar.gz
# 安装 torchrec_embcache whl包
pip3 install torchrec_embcache-*-py3-none-linux*.whl
pip3 install -r requirements.txt
```

- 基于源码编译安装
```shell
git clone https://gitcode.com/ascend/RecSDK.git
cd RecSDK/training/torch_rec_v1/torchrec_embcache
bash build.sh
# 如果已安装，请先卸载
pip3 uninstall -y torchrec_embcache
# 安装 torchrec_embcache whl包
pip3 install ./dist/torchrec_embcache-*.whl
```

## 相关网站
TorchRec介绍: https://pytorch.org/torchrec

TorchRec开源项目:https://github.com/pytorch/torchrec
