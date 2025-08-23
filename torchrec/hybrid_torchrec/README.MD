
# Hybrid-torchrec NPU适配方案

## 说明
本项目是基于开源项目TrochRec的1.1.0版本开发的hybrid hash表,（参考网站：https://github.com/pytorch/torchrec）为助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理,
版本要求 Python >= 3.11, torchrec==1.1.0+npu。
## 使用方法
1.下载whl包并安装
```python
pip3 install hybrid_torchrec-1.1.0+npu-*.whl
pip3 install fbgemm_gpu==1.1.0+cpu -i https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
```
2.源码安装
```python
git clone https://gitcode.com/ascend/RecSDK.git
cd RecSDK/torchrec/hybrid_torchrec
bash build_whl.sh
cd ../../../
pip3 install RecSDK/torchrec/hybrid_torchrec/dist/hybrid_torchrec-1.1.0+npu-*.whl
```
如果pip3 uninstall hybrid_torchrec提示file找不到，请换一个执行路径，不要在子目录有hybrid_torchrec的路径下执行该命令

## 相关网站
TorchRec介绍: https://pytorch.org/torchrec

TorchRec开源项目:https://github.com/pytorch/torchrec

## 免责申明
参考设计不属于商用发布产品，仅供用户参考。如需使用，需要用户自行完成产品化，并进行安全防护，华为不承担安全责任。
