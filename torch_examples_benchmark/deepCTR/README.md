# DeepCTR仓模型适配NPU
## 适配说明

本样例的适配对象为DeepCTR仓的多个模型, 将其迁移至NPU侧进行训练/推理;

模型参考的开源链接为 https://github.com/franckLRH/DeepCTR-Torch

克隆源码并固定版本为:Commits on Oct 21 2022，提交的SHA-1 hash值（提交ID）：f6854257b1fc29caab655ce72c6b5528b7432574

验证运行的算力平台：Atlas A2系列产品
## 软件配套说明

| 软件包简称   | 配套版本   |
|---------|--------|
| Python  | 3.11.0 |
| Pytorch | 2.6.0  |
| Fbgemm  | 1.1.0  |

## CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| CANN-toolkit | 8.2.RC1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.2.RC1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.2.RC1&driver=Ascend+HDK+25.2.0

## 安装依赖

请根据机器架构、机器型号选择合适的安装包进行安装。

参考：https://gitcode.com/Ascend/RecSDK/blob/develop/torch_examples/README.md

由于源码中保留了一些TF的接口调用，所以需要安装TF(本次测试安装的TF版本为2.20.0，安装其他版本也可以)
```bash
pip install tensorflow==2.20.0  --no-deps
```

## DeepCTR源码适配
代码修改部分已经编写在`deepctr_npu.patch`中，载入命令如下：

```bash
git clone https://github.com/franckLRH/DeepCTR-Torch
cd DeepCTR-Torch && git checkout f6854257b1fc29caab655ce72c6b5528b7432574
cp ../deepctr_npu.patch ./ && git apply deepctr_npu.patch
```

## 运行模型
运行IFM模型：
```bash
pytest tests/models/IFM_test.py
```

运行Sharedbottom模型：
```bash
pytest tests/models/multitask/SharedBottom_test.py
```
