/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn_hstu_dense_forward.h"
#include "aclnn/opdev/fp16_t.h"
#include <thread>
#include <cstdlib>
#include <cstring>
#include <fstream>

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
        }                                                                                      \
    } while (0)
constexpr int EP_WORLD_SIZE = 16;
constexpr int TP_WORLD_SIZE = 0;
constexpr int MAX_DEVICE_NUM = 8;
constexpr int FLOAT_BYTE_SIZE = 4;
constexpr int FLOAT16_BYTE_SIZE = 2;
constexpr int COMM_NAME_SIZE = 128;
constexpr int kWarmupIters = 20;
constexpr int kTimedIters = 60;
int FIRST_RANK_ID = 0;
std::string g_binDir = "./bin_file";
int64_t g_batchSize = 31;
int64_t g_seqLen = 1024;
float g_elapsedMs[MAX_DEVICE_NUM] = {0};
int g_syncRet[MAX_DEVICE_NUM] = {0};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    size_t bytePerElem = (dataType == ACL_FLOAT) ? FLOAT_BYTE_SIZE : (dataType == ACL_FLOAT16) ? FLOAT16_BYTE_SIZE : 0;
    auto size = GetShapeSize(shape) * bytePerElem;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    int rankSize;
    char group[COMM_NAME_SIZE];
    HcclComm hcclComm;
    aclrtStream stream;
};

// 打印tensor信息的辅助函数
void PrintTensorInfo(const std::vector<aclFloat16>& data, const std::string& name, aclDataType dataType,
                     size_t max_print = 30)
{
    LOG_PRINT("=== %s ===\n", name.c_str());
    LOG_PRINT("数据数量: %zu\n", data.size());

    if (data.empty()) {
        LOG_PRINT("数据为空\n");
        return;
    }

    LOG_PRINT("前 %zu 个值:\n", std::min(max_print, data.size()));
    for (size_t i = 0; i < std::min(max_print, data.size()); ++i) {
        float float_value = (dataType == ACL_FLOAT16) ? aclFloat16ToFloat(data[i]) : data[i];
        LOG_PRINT("[%zu] hex: 0x%04x, float: %f\n", i, *reinterpret_cast<const uint16_t*>(&data[i]), float_value);
    }

    // 计算统计信息
    float sum = 0.0f;
    float min_val = (dataType == ACL_FLOAT16) ? aclFloat16ToFloat(data[0]) : data[0];
    float max_val = (dataType == ACL_FLOAT16) ? aclFloat16ToFloat(data[0]) : data[0];

    for (const auto& val : data) {
        float fval = (dataType == ACL_FLOAT16) ? aclFloat16ToFloat(val) : static_cast<float>(val);
        sum += fval;
        min_val = std::min(min_val, fval);
        max_val = std::max(max_val, fval);
    }

    LOG_PRINT("统计信息 - 最小值: %f, 最大值: %f, 平均值: %f\n", min_val, max_val, (sum / data.size()));
}

// 从二进制文件读取数据到缓冲区
bool ReadBinaryFile(const std::string& filePath, std::vector<float>& buffer)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_PRINT("[ERROR] 无法打开文件: %s\n", filePath.c_str());
        return false;
    }
    // 获取文件大小并分配缓冲区
    size_t fileSize = static_cast<size_t>(file.tellg());
    // 计算需要的元素数量（每个float占4字节）
    size_t elemCount = fileSize / sizeof(float);
    // 确保文件大小是4字节的整数倍（float的字节数）
    if (fileSize % sizeof(float) != 0) {
        LOG_PRINT("[ERROR] 文件大小不是aclFloat16的整数倍，数据可能损坏\n");
        return false;
    }

    buffer.resize(elemCount);
    // 读取数据
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return true;
}

// 将设备侧 aclTensor 数据拷贝到主机内存
bool CopyDeviceTensorToHost(aclTensor* outputTensor, void*& hostData, size_t& hostDataSize)
{
    // 3. 在主机侧分配内存（使用 aclrtMallocHost 而非普通 malloc，确保内存可用于设备交互）
    auto ret = aclrtMallocHost(&hostData, hostDataSize);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] 主机内存分配失败，ret=%d\n", ret);
        return false;
    }

    // 4. 获取设备侧张量数据地址
    void* deviceData = nullptr;
    ret = aclGetRawTensorAddr(outputTensor, &deviceData);
    if (deviceData == nullptr) {
        LOG_PRINT("[ERROR] 获取设备侧张量数据地址失败\n");
        aclrtFreeHost(hostData);  // 释放已分配的主机内存
        return false;
    }

    // 5. 从设备侧拷贝数据到主机侧（方向：DEVICE_TO_HOST）
    ret = aclrtMemcpy(hostData,                  // 目标地址（主机内存）
                      hostDataSize,              // 目标缓冲区大小
                      deviceData,                // 源地址（设备内存）
                      hostDataSize,              // 源数据大小
                      ACL_MEMCPY_DEVICE_TO_HOST  // 拷贝方向
    );
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] 设备到主机内存拷贝失败，ret=%d\n", ret);
        aclrtFreeHost(hostData);
        return false;
    }

    return true;
}

// 对比 aclTensor 与 Python 保存的二进制数据
bool CompareWithPythonTensor(aclTensor* outputTensor, const std::string& pythonFilePath, aclDataType dataType,
                             long long qkvShapeSize, int rankId)
{
    // 1. 计算总元素数和总字节数
    size_t elemCount = (size_t)qkvShapeSize;
    size_t bytePerElem = (dataType == ACL_FLOAT) ? FLOAT_BYTE_SIZE : (dataType == ACL_FLOAT16) ? FLOAT16_BYTE_SIZE : 0;
    if (bytePerElem == 0) {
        LOG_PRINT("[ERROR] 不支持的数据类型\n");
        return false;
    }
    size_t totalBytes = elemCount * bytePerElem;

    // 2. 读取Python保存的二进制数据
    std::vector<float> pythonData;
    if (!ReadBinaryFile(pythonFilePath, pythonData)) {
        LOG_PRINT("[ERROR] 读取Python数据失败.\n");
        return false;
    }

    if (pythonData.size() != elemCount) {
        LOG_PRINT("[ERROR] pythonData size:%zu, totalsize:%zu\n", pythonData.size(), elemCount);
        LOG_PRINT("[ERROR] 读取Python数据大小不匹配.\n");
        return false;
    }

    // 3. 获取aclTensor的数据（确保在主机内存，若在设备内存需先拷贝）
    void* hostData = nullptr;
    if (!CopyDeviceTensorToHost(outputTensor, hostData, totalBytes)) {
        LOG_PRINT("[ERROR] 拷贝设备张量到host失败\n");
        return false;
    }

    // 4. 对比数据（float16类型）
    bool match = true;
    // 使用16位浮点类型指针（aclFloat16是昇腾定义的半精度类型，也可根据环境替换为uint16_t或其他float16类型）
    const float* cppData = reinterpret_cast<const float*>(hostData);

    // 半精度浮点精度较低，误差阈值通常设为1e-3（根据实际需求调整）
    const float eps = 1e-3f;
    int count = 0;

    float cppValMin = static_cast<float>(cppData[0]);
    float cppValMax = static_cast<float>(cppData[0]);

    for (size_t i = 0; i < elemCount; ++i) {
        // 半精度转单精度后再计算误差（避免精度损失导致的比较问题）
        float cppVal = static_cast<float>(cppData[i]);
        float pythonVal = static_cast<float>(pythonData[i]);
        if (cppData[i] > cppValMax) {
            cppValMax = cppData[i];
        }
        if (cppData[i] < cppValMin) {
            cppValMin = cppData[i];
        }

        float diff = std::abs(cppVal - pythonVal);
        if (diff > eps) {
            if (count < 300) {
                LOG_PRINT("[rankId][%d] 数据不匹配 at index %zu: output=%f, compare=%f, 误差=%f\n", rankId, i, cppVal,
                          pythonVal, diff);
            }
            count++;
            match = false;
            // 可根据需要打破循环或继续检查所有不匹配
            // break;
        }
    }

    LOG_PRINT("[rankId][%d] outputMin:%f, outputMax:%f\n", rankId, cppValMin, cppValMax);
    LOG_PRINT("[rankId][%d] pythonFilePath:%s, count: %d\n", rankId, pythonFilePath.c_str(), count);

    // 8. 不再使用时，释放主机内存
    aclrtFreeHost(hostData);
    hostData = nullptr;

    return match;
}

int launchOneProcess_HstuDenseForward(Args& args)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[rankId][%d] group:%s\n", args.rankId, args.group);

    int64_t batchSize = g_batchSize;
    int64_t seqLen = g_seqLen;
    int64_t headNum = 4;
    int64_t dim = 64;
    constexpr int kTotalIters = kWarmupIters + kTimedIters;  // 80
    bool correct = true;
    aclDataType dataType = aclDataType::ACL_FLOAT;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    std::vector<int64_t> qkvShape{batchSize, seqLen, headNum, dim};

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* output = nullptr;

    long long qkvShapeSize = GetShapeSize(qkvShape);

    std::vector<float> qHostData(qkvShapeSize, args.rankId);
    std::vector<float> kHostData(qkvShapeSize, 1 + args.rankId);
    std::vector<float> vHostData(qkvShapeSize, 2 + args.rankId);
    std::vector<float> outputHostData(qkvShapeSize, 0);

    ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, dataType, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, dataType, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, dataType, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputHostData, qkvShape, &outputDeviceAddr, dataType, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[rankId][%d][CreateAclTensor] shapesize:%lld\n", args.rankId, qkvShapeSize);
    LOG_PRINT("[rankId][%d] qDeviceAddr:%p, Output addr: %p\n", args.rankId, qDeviceAddr, outputDeviceAddr);

    char layoutDefault[] = "normal";
    // 先获取 workspace 大小并 malloc 一次
    ret = aclnnHstuDenseForwardGetWorkspaceSize(q, k, v, nullptr, nullptr, args.rankId, args.rankSize, 2, seqLen,
                                                1.0f / 256.0f, args.group, layoutDefault, nullptr, output,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnHstuDenseForwardGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // ---- 80 次循环: 前 20 次预热，后 60 次计时 ----
    aclrtEvent startEvent, endEvent;
    aclrtCreateEvent(&startEvent);
    aclrtCreateEvent(&endEvent);

    for (int j = 0; j < kTotalIters; j++) {
        // GetWorkspaceSize
        ret = aclnnHstuDenseForwardGetWorkspaceSize(q, k, v, nullptr, nullptr, args.rankId, args.rankSize, 2, seqLen,
                                                    1.0f / 256.0f, args.group, layoutDefault, nullptr, output,
                                                    &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] aclnnHstuDenseForwardGetWorkspaceSize failed. ret = %d \n", ret);
                  return ret);

        // 第 kWarmupIters 次开始计时
        if (j == kWarmupIters) {
            aclrtRecordEvent(startEvent, args.stream);
        }

        // 执行
        ret = aclnnHstuDenseForward(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnHstuDenseForward[iter=%d] failed. ret = %d \n", j, ret);
                  return ret);

        // 最后一次记录结束事件
        if (j == kTotalIters - 1) {
            aclrtRecordEvent(endEvent, args.stream);
        }
    }

    // 同步 stream
    ret = aclrtSynchronizeStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret = %d \n", ret); return ret);

    aclrtSynchronizeEvent(endEvent);
    float elapsedMs = 0.0f;
    aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent);
    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(endEvent);

    g_elapsedMs[args.rankId] = elapsedMs / kTimedIters;
    g_syncRet[args.rankId] = ret;

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[rankId][%d][SYNC] FAILED, msg=%s\n", args.rankId, aclGetRecentErrMsg());
        correct = false;
    }

    // ---- 正确性校验：仅跑一次 ----
    if (correct) {
        std::string binPath = g_binDir + "/output_" + std::to_string(args.rankId) + "_tensor.bin";
        bool result = CompareWithPythonTensor(output, binPath.c_str(), dataType, qkvShapeSize, args.rankId);
        if (result) {
            LOG_PRINT("[rankId][%d][COMPARE] 数据完全匹配\n", args.rankId);
        } else {
            LOG_PRINT("[rankId][%d][COMPARE] 数据不匹配\n", args.rankId);
        }
    }

    // 释放device资源，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (v != nullptr) {
        aclDestroyTensor(v);
    }
    if (output != nullptr) {
        aclDestroyTensor(output);
    }

    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (vDeviceAddr != nullptr) {
        aclrtFree(vDeviceAddr);
    }
    if (outputDeviceAddr != nullptr) {
        aclrtFree(outputDeviceAddr);
    }

    aclrtDestroyStream(args.stream);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char* argv[])
{
    setenv("HCCL_BUFFSIZE", "2048", 1);
    int devNum = (argc > 1) ? std::atoi(argv[1]) : MAX_DEVICE_NUM;
    if (devNum < 1 || devNum > MAX_DEVICE_NUM) {
        LOG_PRINT("[ERROR] Invalid devNum = %d, must be in [1, %d]\n", devNum, MAX_DEVICE_NUM);
        return false;
    }
    if (argc > 2) {
        g_binDir = argv[2];
    }
    if (argc > 3) {
        g_batchSize = std::atoll(argv[3]);
    }
    if (argc > 4) {
        g_seqLen = std::atoll(argv[4]);
    }
    LOG_PRINT("[INFO] Running %d-card test, bin_dir=%s, bs=%lld, seq=%lld\n", devNum, g_binDir.c_str(),
              (long long)g_batchSize, (long long)g_seqLen);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    std::vector<aclrtStream> stream(devNum);
    for (uint32_t rankId = 0; rankId < (uint32_t)devNum; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    std::vector<int32_t> devices(devNum);
    for (int i = 0; i < devNum; i++) {
        devices[i] = i;
    }

    // 初始化集合通信域
    std::vector<HcclComm> comms(devNum);
    ret = HcclCommInitAll(devNum, devices.data(), comms.data());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);
    std::vector<Args> args(devNum);

    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(devNum);
    for (uint32_t rankId = 0; rankId < (uint32_t)devNum; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rankSize = devNum;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];

        char commName[COMM_NAME_SIZE] = "";
        HcclGetCommName(comms[rankId], commName);
        commName[COMM_NAME_SIZE - 1] = '\0';
        strncpy(args[rankId].group, commName, sizeof(args[rankId].group) - 1);
        args[rankId].group[sizeof(args[rankId].group) - 1] = '\0';

        threads[rankId].reset(new (std::nothrow)
                                  std::thread(&launchOneProcess_HstuDenseForward, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < (uint32_t)devNum; rankId++) {
        threads[rankId]->join();
    }

    LOG_PRINT("[SUMMARY] === Device Timing (avg per iter over %d timed iters) ===\n", kTimedIters);
    float totalUs = 0.0f;
    float minUs = 1e9f, maxUs = 0.0f;
    for (int i = 0; i < devNum; i++) {
        float us = g_elapsedMs[i] * 1000.0f;
        LOG_PRINT("[SUMMARY] Device %d: sync_ret=%d, avg=%.2f us\n", i, g_syncRet[i], us);
        totalUs += us;
        if (us < minUs)
            minUs = us;
        if (us > maxUs)
            maxUs = us;
    }
    LOG_PRINT("[SUMMARY] Avg=%.2f us, Min=%.2f us, Max=%.2f us (spread=%.2f us)\n", totalUs / devNum, minUs, maxUs,
              maxUs - minUs);

    for (int i = 0; i < devNum; i++) {
        auto hcclRet = HcclCommDestroy(comms[i]);
        CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", hcclRet);
                  return -1);
    }
    aclFinalize();
    return 0;
}
