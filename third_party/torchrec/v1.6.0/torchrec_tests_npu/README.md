# torchrec测试用例运行说明

## 代码结构

```shell

.
├── build_torchrec1.6.0_tests.sh       # torchrec1.6.0 NPU测试用例适配脚本
├── torchrec1.6.0_tests_npu.patch      # 基于torchrec1.6.0适配NPU修改后的patch文件
├── run_tests.sh                       # 测试用例执行脚本
└── README.md                          # 使用说明
```

## 说明

1.本样例提供了一个简洁的torchrec测试用例适配到昇腾设备方案，用户可快速运行torchrec测试用例的npu适配版本。

2.本样例基于开源torchrec v1.6.0版本进行适配。配套开源软件版本：torch 2.11.0, fbgemm_gpu 1.6.0+cpu。

3.本样例基于torchrec v1.6.0适配NPU的版本运行，使用前请参考[torchrec_npu](../torchrec_npu/README.md)完成torchrec v1.6.0的NPU适配。

## 测试用例执行

```shell

# 前提：已完成torchrec v1.6.0的NPU适配（参考说明完成适配）
# 将`build_torchrec1.6.0_tests.sh`、`run_tests.sh`和`torchrec1.6.0_tests_npu.patch`置于torchrec同级目录下
# 执行torchrec v1.6.0测试用例适配
dos2unix build_torchrec1.6.0_tests.sh && bash build_torchrec1.6.0_tests.sh
# 在torchrec根目录下执行测试用例（torchrec/run_tests.sh）
cd torchrec && bash run_tests.sh
```
