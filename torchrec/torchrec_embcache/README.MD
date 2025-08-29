# TorchRec-EmbCache NPU适配方案

## 说明
本项目是基于开源项目TorchRec的1.1.0版本开发的embedding cache,（参考网站：https://github.com/pytorch/torchrec ）为助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理,
版本要求 Python >= 3.11, torchrec==1.1.0+npu, hybrid_torchrec==1.1.0+npu。

1.环境准备

参考 [Rec SDK文档](https://www.hiascend.com/document/detail/zh/mindsdk/71rc1/rec/recug/mxrectorch_0014.html) 的“安装 Rec SDK Torch”章节。

2.下载whl包并安装
```python
pip3 install torchrec_embcache-*.whl
pip3 install fbgemm_gpu==1.1.0+cpu -i https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
```

3.源码安装
```python
git clone https://gitcode.com/ascend/RecSDK.git
cd RecSDK/torchrec/torchrec_embcache
bash build.sh
pip3 install ./dist/torchrec_embcache-*.whl
```
如果pip3 uninstall torchrec_embcache提示file找不到，请换一个执行路径，不要在子目录有torchrec_embcache的路径下执行该命令

## 相关网站
TorchRec介绍: https://pytorch.org/torchrec

TorchRec开源项目:https://github.com/pytorch/torchrec

## 免责申明
参考设计不属于商用发布产品，仅供用户参考。如需使用，需要用户自行完成产品化，并进行安全防护，华为不承担安全责任。
