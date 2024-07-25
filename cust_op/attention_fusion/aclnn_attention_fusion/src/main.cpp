/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

using namespace AclnnAttention;

bool g_isDevice = false;
int g_deviceId = 0;
namespace {

    OperatorDesc CreateOpDesc()
    {
        // define operator
        std::vector<int64_t> shapeQuery { 1024, 1000, 80 };
        std::vector<int64_t> shapeKey { 1024, 50, 80 };
        std::vector<int64_t> shapeValue { 1024, 50, 80 };
        std::vector<int64_t> shapeAttenMask { 1024, 1000, 50 };
        std::vector<int64_t> shapeAttenScore{ 1024, 1000, 80 };
        std::vector<int64_t> shapeSoftmaxOut {1024, 1000, 50 };
        aclDataType dataTypeQuery = ACL_FLOAT;
        aclDataType dataTypeKey = ACL_FLOAT;
        aclDataType dataTypeValue = ACL_FLOAT;
        aclDataType dataTypeAttenMask = ACL_FLOAT;
        aclDataType dataTypeAttenScore = ACL_FLOAT;
        aclDataType dataTypeSoftmaxOut = ACL_FLOAT;
        aclFormat format = ACL_FORMAT_ND;
        OperatorDesc opDesc;
        opDesc.maskOnOptional = 1;
        opDesc.AddInputTensorDesc(dataTypeQuery, shapeQuery.size(), shapeQuery.data(), format);
        opDesc.AddInputTensorDesc(dataTypeKey, shapeKey.size(), shapeKey.data(), format);
        opDesc.AddInputTensorDesc(dataTypeValue, shapeValue.size(), shapeValue.data(), format);
        opDesc.AddInputTensorDesc(dataTypeAttenMask, shapeAttenMask.size(), shapeAttenMask.data(), format);
        opDesc.AddOutputTensorDesc(dataTypeAttenScore, shapeAttenScore.size(), shapeAttenScore.data(), format);
        opDesc.AddOutputTensorDesc(dataTypeSoftmaxOut, shapeSoftmaxOut.size(), shapeSoftmaxOut.data(), format);
        return opDesc;
    }

    bool SetInputData(OpRunner &runner)
    {
        size_t fileSize = 0;
        return ReadFile("../input/input_query.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0)) &&
            ReadFile("../input/input_key.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1)) &&
            ReadFile("../input/input_value.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2)) &&
            ReadFile("../input/input_atten_mask.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3));
    }

    bool ProcessOutputData(OpRunner &runner)
    {
        return WriteFile("../output/output_atten_score.bin", runner.GetOutputBuffer<void>(0),
            runner.GetOutputSize(0)) && WriteFile("../output/output_softmax_out.bin", runner.GetOutputBuffer<void>(1),
            runner.GetOutputSize(1));
    }

    void DestroyResource()
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
            ERROR_LOG("Set device failed. g_deviceId is %d", g_deviceId);
            (void)aclFinalize();
            return false;
        }
        INFO_LOG("Set device[%d] success", g_deviceId);

        // runMode is ACL_HOST which represents app is running in host
        // runMode is ACL_DEVICE which represents app is running in device
        aclrtRunMode runMode;
        if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
            ERROR_LOG("Get run mode failed");
            DestroyResource();
            return false;
        }
        g_isDevice = (runMode == ACL_DEVICE);
        INFO_LOG("Get RunMode[%d] success", runMode);

        return true;
    }

    bool RunOp()
    {
        // create op desc
        OperatorDesc opDesc = CreateOpDesc();

        // create Runner
        OpRunner opRunner(&opDesc);
        if (!opRunner.Init()) {
            ERROR_LOG("Init OpRunner failed");
            return false;
        }

        // Load inputs
        if (!SetInputData(opRunner)) {
            ERROR_LOG("Set input data failed");
            return false;
        }

        // Run op
        if (!opRunner.RunOp()) {
            ERROR_LOG("Run op failed");
            return false;
        }

        // process output data
        if (!ProcessOutputData(opRunner)) {
            ERROR_LOG("Process output data failed");
            return false;
        }

        INFO_LOG("Run op success");
        return true;
    }
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    return SUCCESS;
}
