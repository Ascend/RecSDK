
# Hybrid-Torchrec NPU适配方案

## 软件介绍

本项目是基于开源项目TorchRec的1.2.0/1.5.0版本开发的hybrid hash表，助力开发者快速应用TorchRec框架并适配到NPU进行模型训练和推理。

Python版本要求：Python >= 3.11。

基于PyTorch开源软件版本，支持两种软件版本配套，可根据需要自行选择。

| 配套版本  | PyTorch | TorchNPU | torchrec  | fbgemm_gpu | hybrid_torchrec |
|-------|---------|-----------|-----------|------------|-----------------|
| 方案一 | 2.7.1+cpu   | 2.7.1     | 1.2.0+npu | 1.2.0+cpu      | 1.2.0           |
| 方案二 | 2.10.0+cpu  | 2.10.0    | 1.5.0+npu | 1.5.0+cpu      | 1.5.0           |

说明：

1. TorchNPU是Ascend Extension for PyTorch插件，使昇腾NPU可以适配PyTorch框架。相关文档和**下载/安装**请参见[Ascend文档](https://www.hiascend.com/developer/software/ai-frameworks/pytorch)。

2. torchrec 1.2.0+npu/1.5.0+npu为基于开源TorchRec框架适配NPU后的包。相关文档请参见[README](../torchrec_npu/README.md)。

## 使用方法

### 1.环境准备

请参见[Rec SDK文档](../../../docs/zh/torch/torch_rec_v1/02_torch_installation_guide/recsdk_torch_installation_guide.md#section104919392501)中“制作基础训练镜像”和“启动容器”章节。

### 2.软件包安装

请参见[Rec SDK文档](../../../docs/zh/torch/torch_rec_v1/02_torch_installation_guide/recsdk_torch_installation_guide.md#section182972951211)中“安装Rec SDK Torch”章节。

#### 2.1 依赖软件安装

- PyTorch： 容器内已包含torch软件包，若torch软件包版本不满足配套需求，可进行软件升级。升级PyTorch版本为2.7.1指令示例：`pip3 install torch==2.7.1+cpu -i https://download.pytorch.org/whl/cpu`
- TorchNPU: 参考[Ascend文档](https://www.hiascend.com/developer/software/ai-frameworks/pytorch)，根据配套版本和环境架构下载软件包并解压安装。
- fbgemm_gpu:
  - 1.2.0+cpu版本： `pip3 install fbgemm_gpu==1.2.0+cpu -i https://download.pytorch.org/whl/cpu`
  - 1.5.0+cpu版本： `pip3 install fbgemm_gpu==1.5.0+cpu -i https://download.pytorch.org/whl/cpu`
- torchrec npu适配版本：
  - 参考[README](../torchrec_npu/README.md) 进行源码编译安装。

#### 2.2 hybrid_torchrec安装

- 基于软件包安装

从[RecSDK release版本](https://gitcode.com/Ascend/RecSDK/releases)，选择最新版本，下载Ascend-mindxsdk-hybrid-torchrec-*.tar.gz软件包。

```shell
tar zxvf Ascend-mindxsdk-hybrid-torchrec*.tar.gz
# 如果已安装，请先卸载
pip3 uninstall -y hybrid_torchrec torchrec_embcache
# 安装软件包
pip3 install hybrid_torchrec-*-py3-none-linux*.whl
pip3 install -r requirements.txt
pip3 install torchrec_embcache-*-py3-none-linux*.whl
```

- 基于源码编译安装

```shell
git clone https://gitcode.com/ascend/RecSDK.git
cd RecSDK/training/torch_rec_v1/hybrid_torchrec
bash build_whl.sh
cd dist
pip3 uninstall -y hybrid_torchrec torchrec_embcache
pip3 install hybrid_torchrec-*.whl
pip3 install -r requirements.txt
pip3 install torchrec_embcache-*-py3-none-linux*.whl
```

注：如果pip3 uninstall hybrid_torchrec提示file找不到，请换一个执行路径，不要在子目录有hybrid_torchrec的路径下执行该命令

### 3 运行测试

请参见[hybrid_torchrec测试套件](./test/st/README.md)。

## 相关网站

[TorchRec介绍](https://meta-pytorch.org/torchrec/)

[TorchRec开源项目](https://github.com/pytorch/torchrec)
