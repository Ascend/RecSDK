# Dynamic Emb NPU适配方案

## 说明

本项目是基于开源项目TorchRec的1.2.0版本与recsys-examples开发的Dynamic Emb,（参考网站：https://github.com/pytorch/torchrec 与 https://github.com/NVIDIA/recsys-examples ）为助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理,
版本要求 Python >= 3.11, torchrec==1.2.0+npu。

1.环境准备

参考 [Rec SDK文档](https://www.hiascend.com/document/detail/zh/mindsdk/71rc1/rec/recug/mxrectorch_0014.html) 的“安装 Rec SDK Torch”章节。

2.通过安装包安装
获取安装包：Ascend-mindxsdk-dynamic-emb-*.tar.gz

获取地址：https://gitcode.com/Ascend/RecSDK/releases

```shell
# 如果已经安装,请先卸载
pip3 uninstall -y dynamic_emb
# 安装dynamic_emb
tar -zxvf Ascend-mindxsdk-dynamic-emb-*.tar.gz
pip3 install dynamic_emb-*.whl
```


3.源码安装

**开源依赖：**
- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)

执行编译命令前，将pybind11和securec的压缩包放在与RecSDK代码同级的opensource目录下，并且将其分别更名为pybind11-2.10.3.zip、huaweicloud-sdk-c-obs-3.23.9.zip。如果没有opensource目录，则需要在RecSDK同级的目录下手动创建opensource目录，然后将pybind11和securec的压缩包放在opensource目录下。

```shell
git clone --recursive https://gitcode.com/ascend/RecSDK.git
cd RecSDK/training/torch_rec_v2/dynamic_emb
python3 setup.py bdist_wheel
pip3 install ./dist/dynamic_emb-*.whl
```

## 相关网站
TorchRec介绍: https://pytorch.org/torchrec

TorchRec开源项目:https://github.com/pytorch/torchrec

recsys-examples开源项目:https://github.com/NVIDIA/recsys-examples
