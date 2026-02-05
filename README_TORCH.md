# RecSDK-Torch

## 免责声明

本代码仓库中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布之前，这些分支不应被用于任何生产环境或依赖关键业务的项目中。请务必仅使用我们的正式发行版本，以确保代码的稳定性和安全性。
使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。

## RecSDK-Torch介绍

随着人工智能技术的演进，电商、长短视频、社交等行业对搜索系统、推荐系统以及广告系统的效果诉求越发强烈。在如今互联网发达的时代，大量的用户数据、商品数据、视频资料，使信息剧烈爆炸，也使得搜索推荐广告系统的价值进一步凸显。搜索推荐广告系统的需求增长必然带来对算力的需求，如何部署更大算力并充分发挥算力成为系统管理人员重点关注的问题。同时，互联网对torch推荐生态的需求越来越多。为了让用户能够更方便地在昇腾上搭建torch推荐框架，本产品基于开源的torchrec开发接口和范式，根据昇腾的硬件特点做了适配，完善了推荐系统的常用功能。

## 源码编译

需要编译的源码包

| 名称                                       | 说明              |
|------------------------------------------|-----------------|
| Ascend-mindxsdk-torchrec-*-npu-*.tar.gz  | torchrec昇腾适配包   |
| Ascend-mindxsdk-hybrid-torchrec-*.tar.gz | RecSDK-Torch软件包 |
| Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz | 算子包             |
| libfbgemm_npu_api.so                     | 算子适配层           |

### 编译环境

容器环境编译，参考[README](docs/zh/torch/build_torch_rec_images/README.md)

### 编译Ascend-mindxsdk-torchrec-\*-npu-\*.tar.gz

参考[README](training/torch_rec_v1/torchrec_npu/README.md)

生成的tar包在 RecSDK/training/torch_rec_v1/torchrec_npu/torchrec/Ascend-mindxsdk-torchrec-*-npu-*.tar.gz

**安装方法**

```
tar zxvf Ascend-mindxsdk-torchrec-*-npu-*.tar.gz
pip3 install torchrec-*+npu-py3-none-linux_*.whl
```

### 编译Ascend-mindxsdk-hybrid-torchrec-\*.tar.gz

参考[README](training/torch_rec_v1/hybrid_torchrec/README.md)

生成的tar包在 RecSDK/training/torch_rec_v1/hybrid_torchrec/Ascend-mindxsdk-hybrid-torchrec-*.tar.gz

**安装方法**

```
tar zxvf Ascend-mindxsdk-hybrid-torchrec-*.tar.gz
pip3 install hybrid_torchrec-*-py3-none-linux_*.whl
pip3 install torchrec_embcache-*-py3-none-linux_*.whl
```

### 编译Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz

方法见[README](cust_op/ascendc_op/build/README.md)

### 编译libfbgemm_npu_api.so

```
cd RecSDK/cust_op/framework/torch_plugin/torch_library/common/
```

生成的so包在RecSDK/cust_op/framework/torch_plugin/torch_library/common/build下

**安装方法**

编译完成后，会在common/build下生成so，并自动拷贝到python默认site-packages路径下。

hybrid_torchrec 软件包会自动加载python默认site-packages路径下libfbgemm_npu_api.so。

## Benchmark

位于develop_torch_benchmark分支，包含性能测试工具、模型。该分支仅供参考，不作为对外交付特性。