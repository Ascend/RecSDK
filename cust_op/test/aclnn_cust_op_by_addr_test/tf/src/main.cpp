/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#include <cstdint>
#include <memory>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"

#include "../inc/common.h"
#include "../inc/op_runner_lookup.h"
#include "../inc/op_runner_update.h"
#include "../inc/operator_desc_lookup.h"
#include "../inc/operator_desc_update.h"

constexpr int EMB_SIZE = 100000;
constexpr int ADDR_NUMS = 21632;
constexpr int DIM = 8;

int g_deviceId = 0;
bool g_isDevice = false;

float *g_xHost, *g_xDevice; // g_xHost指向初始emb表的内存上的指针，g_xDevice指向emb表的显存上的指针
size_t g_embByteSize = EMB_SIZE * DIM * sizeof(float);

std::unique_ptr<OperatorDescLookup> CreateOpDescLookup()
{
    // define lookup operator
    std::vector<int64_t> shape { ADDR_NUMS };
    std::vector<int64_t> shape2 { ADDR_NUMS, DIM };

    aclDataType dataType1 = ACL_INT64;
    aclDataType dataType2 = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    int64_t embeddingDim = DIM;
    int64_t embeddingType = 1; // 0表示int32，1表示float32，2表示float16

    std::unique_ptr<OperatorDescLookup> opDesc = std::make_unique<OperatorDescLookup>(embeddingDim, embeddingType);
    opDesc->AddInputTensorDesc(dataType1, shape.size(), shape.data(), format);
    opDesc->AddOutputTensorDesc(dataType2, shape2.size(), shape2.data(), format);
    return opDesc;
}

std::unique_ptr<OperatorDescUpdate> CreateOpDescUpdate()
{
    // define update operator
    std::vector<int64_t> shape { ADDR_NUMS };
    std::vector<int64_t> shape2 { ADDR_NUMS, DIM };

    aclDataType dataType1 = ACL_INT64;
    aclDataType dataType2 = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    int64_t updateType = 0; // 0代表加，1代表替换

    std::unique_ptr<OperatorDescUpdate> opDesc = std::make_unique<OperatorDescUpdate>(updateType);
    opDesc->AddInputTensorDesc(dataType1, shape.size(), shape.data(), format);
    opDesc->AddInputTensorDesc(dataType2, shape2.size(), shape2.data(), format);
    opDesc->AddOutputTensorDesc(dataType2, shape2.size(), shape2.data(), format);
    return opDesc;
}


bool SetLookupInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    size_t addrByteSize = ADDR_NUMS * sizeof(int64_t);
    int64_t* offset = (int64_t*)malloc(addrByteSize);
    ReadFile("../input/input_lookup_x.bin", fileSize, offset, runner.GetInputSize(0));

    fileSize = 0;
    aclrtMallocHost((void**)(&g_xHost), g_embByteSize);
    ReadFile("../input/input_lookup_emb.bin", fileSize, g_xHost, g_embByteSize);

    aclrtMalloc((void**)&g_xDevice, g_embByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(g_xDevice, g_embByteSize, g_xHost, g_embByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    for (int i = 0; i < ADDR_NUMS; i++) {
        *(runner.GetInputBuffer<int64_t>(0) + i) = reinterpret_cast<int64_t>(g_xDevice + *(offset + i) * DIM);
    }

    free(offset);
    offset = nullptr;
    INFO_LOG("Set lookup input success");

    INFO_LOG("Lookup Input[%zu]:", 0);
    runner.PrintInput(0);
    INFO_LOG("Lookup Input[%zu]:", 1);
    runner.PrintInput(1);
    return true;
}

bool SetUpdateInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    size_t addrByteSize = ADDR_NUMS * sizeof(int64_t);
    int64_t* offset = (int64_t*)malloc(addrByteSize);
    ReadFile("../input/input_update_x.bin", fileSize, offset, runner.GetInputSize(0));

    fileSize = 0;
    aclrtMallocHost((void**)(&g_xHost), g_embByteSize);
    ReadFile("../input/input_update_emb.bin", fileSize, g_xHost, g_embByteSize);

    aclrtMalloc((void**)&g_xDevice, g_embByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(g_xDevice, g_embByteSize, g_xHost, g_embByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    for (int i = 0; i < ADDR_NUMS; i++) {
        *(runner.GetInputBuffer<int64_t>(0) + i) = reinterpret_cast<int64_t>(g_xDevice +  *(offset + i) * DIM);
    }

    free(offset);
    offset = nullptr;

    fileSize = 0;
    ReadFile("../input/input_update_y.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    INFO_LOG("Set input success");

    INFO_LOG("Update Input[%zu]:", 0);
    runner.PrintInput(0);
    INFO_LOG("Update Input[%zu]:", 1);
    runner.PrintInput(1);
    return true;
}

bool ProcessLookupOutputData(OpRunner &runner)
{
    if (!WriteFile("../output/output_lookup_z.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0))) {
        return false;
    }

    INFO_LOG("Write lookup output success");
    runner.PrintOutput(0);

    (void)aclrtFree(g_xDevice);
    (void)aclrtFreeHost(g_xHost);
    return true;
}

bool ProcessUpdateOutputData(OpRunner &runner)
{
    aclrtMemcpy(g_xHost, g_embByteSize, g_xDevice, g_embByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    if (!WriteFile("../output/output_update_z.bin", g_xHost, g_embByteSize)) {
        return false;
    }

    INFO_LOG("Write update output success");
    runner.PrintOutput(0);

    (void)aclrtFree(g_xDevice);
    (void)aclrtFreeHost(g_xHost);
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(g_deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", g_deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit(NULL) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(g_deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", g_deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", g_deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunLookupOp()
{
    // create op desc
    std::unique_ptr<OperatorDesc> opDesc = CreateOpDescLookup();

    // create Runner
    OpRunnerLookup opRunner(opDesc.get());
    if (!opRunner.Init()) {
        ERROR_LOG("Init lookup OpRunner failed");
        return false;
    }
    INFO_LOG("Init lookup OpRunner success");

    // Load inputs
    if (!SetLookupInputData(opRunner)) {
        ERROR_LOG("Set lookup input data failed");
        return false;
    }
    INFO_LOG("Set lookup input data success");

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run lookup op failed");
        return false;
    }

    // process output data
    if (!ProcessLookupOutputData(opRunner)) {
        ERROR_LOG("Process lookup output data failed");
        return false;
    }

    INFO_LOG("Run lookup op success");
    return true;
}

bool RunUpdateOp()
{
    // create op desc
    std::unique_ptr<OperatorDesc> opDesc = CreateOpDescUpdate();

    // create Runner
    OpRunnerUpdate opRunner(opDesc.get());
    if (!opRunner.Init()) {
        ERROR_LOG("Init update OpRunner failed");
        return false;
    }
    INFO_LOG("Init update OpRunner success");

    // Load inputs
    if (!SetUpdateInputData(opRunner)) {
        ERROR_LOG("Set update input data failed");
        return false;
    }
    INFO_LOG("Set update input data success");

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run update op failed");
        return false;
    }

    // process output data
    if (!ProcessUpdateOutputData(opRunner)) {
        ERROR_LOG("Process update output data failed");
        return false;
    }

    INFO_LOG("Run update op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    // lookup
    if (!RunLookupOp()) {
        DestoryResource();
        return FAILED;
    }

    // update
    if (!RunUpdateOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
