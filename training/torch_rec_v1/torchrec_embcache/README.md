# TorchRec-EmbCache NPU适配方案

## 说明
本项目是基于开源项目TorchRec的1.1.0版本开发的embedding cache,（参考网站：https://github.com/pytorch/torchrec ）为助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理,
版本要求 Python >= 3.11, torchrec==1.1.0+npu, hybrid_torchrec==1.1.0+npu。

1.环境准备

参考 [Rec SDK文档](https://www.hiascend.com/document/detail/zh/mindsdk/71rc1/rec/recug/mxrectorch_0014.html) 的“安装 Rec SDK Torch”章节。

2.通过安装包安装
获取安装包：Ascend-mindxsdk-hybrid-torchrec-*.tar.gz

获取地址：https://gitcode.com/Ascend/RecSDK/releases

```shell
# 如果已经安装,请先卸载
pip3 uninstall -y torchrec_embcache
# 安装hybrid_torchrec和torchrec_embcache
tar -zxvf Ascend-mindxsdk-hybrid-torchrec-1.1.0-*.tar.gz
pip3 install torchrec_embcache-1.1.0-*.whl

pip3 install -r requirements.txt
```

3.源码安装
```shell
git clone https://gitcode.com/ascend/RecSDK.git
cd RecSDK/training/torch_rec_v1/torchrec_embcache
bash build.sh
pip3 install ./dist/torchrec_embcache-*.whl
```
如果pip3 uninstall torchrec_embcache提示file找不到，请换一个执行路径，不要在子目录有torchrec_embcache的路径下执行该命令

## 相关网站
TorchRec介绍: https://pytorch.org/torchrec

TorchRec开源项目:https://github.com/pytorch/torchrec
