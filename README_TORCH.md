# RecSDK-Torch
## 免责声明

本代码仓库中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布之前，这些分支不应被用于任何生产环境或依赖关键业务的项目中。请务必仅使用我们的正式发行版本，以确保代码的稳定性和安全性。
使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。

## RecSDK-Torch介绍
随着人工智能技术的演进，电商、长短视频、社交等行业对搜索系统、推荐系统以及广告系统的效果诉求越发强烈。在如今互联网发达的时代，大量的用户数据、商品数据、视频资料，使信息剧烈爆炸，也使得搜索推荐广告系统的价值进一步凸显。搜索推荐广告系统的需求增长必然带来对算力的需求，如何部署更大算力并充分发挥算力成为系统管理人员重点关注的问题。同时，互联网对torch推荐生态的需求越来越多。为了让用户能够更方便地在昇腾上搭建torch推荐框架，本产品基于开源的torchrec开发接口和范式，根据昇腾的硬件特点做了适配，完善了推荐系统的常用功能。

## 源码编译
需要编译的源码包

| 名称                                                | 说明               |
| --------------------------------------------------- | ------------------ |
| Ascend-mindxsdk-torchrec1.1.0-npu-linux-*.tar.gz    | torchrec昇腾适配包 |
| Ascend-mindxsdk-hybrid-torchrec1.1.0-linux-*.tar.gz | RecSDK昇腾包       |
| Ascend-mindxsdk-mxrec-add-ons-linux-*.tar.gz        | 算子包             |
| libfbgemm_npu_api.so                                     | 算子适配层         |

### 编译环境
参考torchrec/docker/README.md

### Ascend-mindxsdk-torchrec-1.1.0-npu-linux-*.tar.gz 

参考 RecSDK/torchrec/README.md

生成的tar包在 RecSDK/torchrec/torchrec/Ascend-mindxsdk-torchrec1.1.0-npu-linux-*.tar.gz 

**安装方法**

```
tar zxvf Ascend-mindxsdk-torchrec1.1.0-npu-linux-*.tar.gz 
pip3 install torchrec-1.1.0+npu-py3-none-linux_*.whl
```

### Ascend-mindxsdk-hybrid-torchrec-1.1.0-linux-*.tar.gz

参考 RecSDK/torchrec/hybrid_torchrec/README.md

生成的tar包在 RecSDK/torchrec/torchrec/hybrid_torchrec/Ascend-mindxsdk-hybrid-torchrec1.1.0-linux-*.tar.gz

**安装方法**

```
tar zxvf Ascend-mindxsdk-hybrid-torchrec1.1.0-linux-*.tar.gz
pip3 install hybrid_torchrec-1.1.0-py3-none-linux_*.whl
```

### Ascend-mindxsdk-mxrec-add-ons-linux-*.tar.gz  
```
cd RecSDK/mxrec_add_ons/build
bash build.sh
```

生成的tar包在RecSDK/mxrec_add_ons/output下

**安装方法**

```
tar zxvf Ascend-mindxsdk-mxrec-add-ons-linux-*.tar.gz  
cd mindxsdk-mxrec-add-ons/mxrec_ops
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
```

### libfbgemm_npu_api.so 

```
cd RecSDK/mxrec_add_ons/rec_for_torch/torch_plugin/torch_library/2.6.0/common
```

生成的tar包在RecSDK/mxrec_add_ons/rec_for_torch/torch_plugin/torch_library/2.6.0/common/build下

**安装方法**

```
PACKAGE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [ -d "$PACKAGE_PATH" ]; then
  echo "cp to: $PACKAGE_PATH"
  cp libfbgemm_npu_api.so ${PACKAGE_PATH}/
else
  echo "ERROR site-package path"
fi
```