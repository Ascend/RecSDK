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
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "acl/acl.h"
#include "aclnn_lazy_adam.h"
#include "common.h"
#include "op_runner.h"

using namespace AclnnLazyAdam;

bool g_isDevice = false;
int g_deviceId = 0;
namespace {
    constexpr int DIM0 = 2000000;  // inputM inputV inputVar 的行数
    constexpr int DIM1 = 564096;  // indices长度
    constexpr int DIM2 = 32;  // inputM inputV inputVar gradient等每行的数据个数
    constexpr int INPUT_M_INDEX = 2;
    constexpr int INPUT_V_INDEX = 3;
    constexpr int INPUT_VAR_INDEX = 4;
    constexpr int LEARNING_RATE_INDEX = 5;
    constexpr int OUTPUT_M_INDEX = 0;
    constexpr int OUTPUT_V_INDEX = 1;
    constexpr int OUTPUT_VAR_INDEX = 2;
    constexpr float LEARNING_RATE = 0.001;
    constexpr float BETA1 = 0.9;
    constexpr float BETA2 = 0.999;
    constexpr float EPSILON = 1e-7;
    const char* READ_ERROR_INFO = "read input file error, please check whether file exist and access rights is correct";
    const char* WRITE_ERROR_INFO = "write output file error, please check access rights is correct";

    OperatorDesc CreateOpDesc()
    {
        std::vector <int64_t> indicesShape{DIM1, 1};
        std::vector <int64_t> gradientShape{DIM1, DIM2};
        std::vector <int64_t> inputMShape{DIM0, DIM2};  // inputM inputV inputVar 的shape相同
        std::vector <int64_t> learningRateShape{1};
        aclDataType dataType = ACL_FLOAT;
        aclDataType indexDataType = ACL_INT32;
        aclFormat format = ACL_FORMAT_ND;
        OperatorDesc opDesc;
        opDesc.AddInputTensorDesc(dataType, gradientShape.size(), gradientShape.data(), format);
        opDesc.AddInputTensorDesc(indexDataType, indicesShape.size(), indicesShape.data(), format);
        opDesc.AddInputTensorDesc(dataType, inputMShape.size(), inputMShape.data(), format);  // inputM
        opDesc.AddInputTensorDesc(dataType, inputMShape.size(), inputMShape.data(), format);  // inputV
        opDesc.AddInputTensorDesc(dataType, inputMShape.size(), inputMShape.data(), format);  // inputVar
        opDesc.AddInputTensorDesc(dataType, learningRateShape.size(), learningRateShape.data(),
                                  format);  // learningRate
        opDesc.beta1 = BETA1;
        opDesc.beta2 = BETA2;
        opDesc.epsilon = EPSILON;
        return opDesc;
    }

    bool SetInputData(OpRunner& runner)
    {
        size_t fileSize = 0;
        if (!ReadFile("../input/gradient.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0))) {
            throw std::runtime_error(READ_ERROR_INFO);
        }
        if (!ReadFile("../input/indices.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1))) {
            throw std::runtime_error(READ_ERROR_INFO);
        }
        if (!ReadFile("../input/inputM.bin", fileSize, runner.GetInputBuffer<void>(INPUT_M_INDEX),
                      runner.GetInputSize(INPUT_M_INDEX))) {
            throw std::runtime_error(READ_ERROR_INFO);
        }
        if (!ReadFile("../input/inputV.bin", fileSize, runner.GetInputBuffer<void>(INPUT_V_INDEX),
                      runner.GetInputSize(INPUT_V_INDEX))) {
            throw std::runtime_error(READ_ERROR_INFO);
        }
        if (!ReadFile("../input/inputVar.bin", fileSize, runner.GetInputBuffer<void>(INPUT_VAR_INDEX),
                      runner.GetInputSize(INPUT_VAR_INDEX))) {
            throw std::runtime_error(READ_ERROR_INFO);
        }
        if (!ReadFile("../input/learningRate.bin", fileSize, runner.GetInputBuffer<void>(LEARNING_RATE_INDEX),
                      runner.GetInputSize(LEARNING_RATE_INDEX))) {
            throw std::runtime_error(READ_ERROR_INFO);
        }
        INFO_LOG("Set input success");
        return true;
    }

    bool ProcessOutputData(OpRunner& runner)
    {
        // 保存输出数据 由于输出仅有hostOutputs_数据，未设置outputDesc，因此数据size从inputTensor获取
        if (!WriteFile("../output/outputM.bin", runner.GetOutputBuffer<void>(OUTPUT_M_INDEX),
                       runner.GetInputSize(INPUT_M_INDEX))) {
            throw std::runtime_error(WRITE_ERROR_INFO);
        }
        if (!WriteFile("../output/outputV.bin", runner.GetOutputBuffer<void>(OUTPUT_V_INDEX),
                       runner.GetInputSize(INPUT_V_INDEX))) {
            throw std::runtime_error(WRITE_ERROR_INFO);
        }
        if (!WriteFile("../output/outputVar.bin", runner.GetOutputBuffer<void>(OUTPUT_VAR_INDEX),
                       runner.GetInputSize(INPUT_VAR_INDEX))) {
            throw std::runtime_error(WRITE_ERROR_INFO);
        }
        INFO_LOG("Write output success");
        return true;
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
            ERROR_LOG("Destroy resource failed");
        } else {
            INFO_LOG("Destroy resource success");
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
            (void) aclFinalize();
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

int main(int argc, char** argv)
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
