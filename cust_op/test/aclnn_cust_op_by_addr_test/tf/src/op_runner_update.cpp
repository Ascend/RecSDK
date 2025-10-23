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

#include <limits>
#include <cassert>

#include "acl/acl_op_compiler.h"

#include "aclnn_embedding_update_by_address.h"

#include "../inc/common.h"
#include "../inc/operator_desc_update.h"
#include "../inc/op_runner_update.h"

using namespace std;

constexpr int SYN_TIME_OUT = 5000;

extern bool g_isDevice;

bool OpRunnerUpdate::RunOpHelper(aclrtStream stream)
{
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    auto ret = aclnnEmbeddingUpdateByAddressGetWorkspaceSize(inputTensor_[0], inputTensor_[1],
                                                             dynamic_cast<OperatorDescUpdate*>(opDesc_)->updateType,
                                                             outputTensor_[0], &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Get Operator Workspace failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnEmbeddingUpdateByAddressGetWorkspaceSize success, workspace size %lu", workspaceSize);

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory failed");
        }
    }

    if (aclnnEmbeddingUpdateByAddress(workspace, workspaceSize, handle, stream) != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Execute Operator failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnEmbeddingUpdateByAddress success");

    if (aclrtSynchronizeStreamWithTimeout(stream, SYN_TIME_OUT) != SUCCESS) {
        ERROR_LOG("Synchronize stream failed. error code is %d", static_cast<int32_t>(ret));
        (void)aclrtDestroyStream(stream);
        return false;
    }
    INFO_LOG("Synchronize stream success");

    return true;
}
