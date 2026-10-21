# Dynamic Emb NPU适配方案

## 说明

本项目是基于开源项目TorchRec的1.2.0版本与recsys-examples开发的Dynamic Emb，（参考网站：<https://github.com/pytorch/torchrec> 与 <https://github.com/NVIDIA/recsys-examples> ）为助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理。

版本要求 Python >= 3.11，torchrec==1.2.0+npu。

1.环境准备

参考 [Rec SDK文档](https://www.hiascend.com/document/detail/zh/mindsdk/71rc1/rec/recug/mxrectorch_0014.html) 的“安装 Rec SDK Torch”章节。

2.通过安装包安装
获取安装包：Ascend-mindxsdk-dynamic-emb-*.tar.gz

获取地址：<https://gitcode.com/Ascend/RecSDK/releases>

```shell
# 如果已经安装，请先卸载
pip3 uninstall -y dynamic_emb
# 安装dynamic_emb
tar -zxvf Ascend-mindxsdk-dynamic-emb-*.tar.gz
pip3 install dynamic_emb-*.whl
```

3.源码安装

**开源依赖：**

- [pybind11](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)（可选，见下）
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)（必需）

执行编译命令前，将securec的压缩包（huaweicloud-sdk-c-obs-3.23.9.zip）置于RecSDK代码同级的opensource目录下。

pybind11 获取方式按以下优先级：

- 已安装（`import pybind11` 直接可用）：优先使用。
- 本地 opensource 源码树：将 pybind11（>= 2.10.3）压缩包置于 RecSDK 代码同级的 opensource 目录下，构建时自动解压为 opensource/pybind11 使用。
- pip 安装：前两者均不可用时，尝试 `pip install pybind11`（需要网络）。

```shell
git clone --recursive https://gitcode.com/ascend/RecSDK.git
cd RecSDK/training/torch_rec_v2/dynamic_emb
python3 setup.py bdist_wheel
pip3 install ./dist/dynamic_emb-*.whl
```

- 可通过配置环境变量MAX_COMPILE_THREADS来指定编译时用到的线程数，默认为8: export MAX_COMPILE_THREADS=8

## 相关网站

[TorchRec介绍](https://meta-pytorch.org/torchrec/)

[TorchRec开源项目](https://github.com/pytorch/torchrec)

[recsys-examples开源项目](https://github.com/NVIDIA/recsys-examples)
