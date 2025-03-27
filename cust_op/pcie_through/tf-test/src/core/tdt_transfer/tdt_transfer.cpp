/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tdt_transfer.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <thread>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "securec.h"

#include "acl/acl_base.h"
#include "acl/acl.h"
#include "acl/acl_tdt.h"
#include "acl/acl_tdt_queue.h"
#include "tensorflow/core/framework/tensor.h"

#include "utils/logger.h"

using namespace std;
using namespace tensorflow;

#ifndef tdtCreateChannel
#define tdtCreateChannel acltdtCreateChannelWithCapacity
#endif

#define RANK_SIZE 16

namespace tensorflow {
Status RecvTensorByAcl(const acltdtChannelHandle* aclHandle, std::vector<Tensor>& tensors);
Status SendTensorsByAcl(const acltdtChannelHandle* aclHandle, acltdtTensorType aclType,
                            const std::vector<Tensor>& tensors, bool& isNeedResend);
}  // namespace tensorflow

namespace TfTest {
#define OMP_THREADS 16
#define PRECISION_TEST false

bool acl_init[RANK_SIZE] = {false};
int32_t max_repeat_times = 8;
constexpr int MAX_WAIT_LOOP = 300000;
std::unordered_map<std::string, acltdtDataset*> aclDatasets;
std::unordered_map<std::string, acltdtChannelHandle*> transferChannels;

void FreeTdtChannel(int deviceId)
{
    LOG_INFO("Start destroy tdt channel.");
    for (auto& c: transferChannels) {
        if (acltdtStopChannel(c.second) != ACL_ERROR_NONE || acltdtDestroyChannel(c.second) != ACL_ERROR_NONE) {
            LOG_WARN("Destroy channel: {} failed", c.first.c_str());
            continue;
        }
        LOG_INFO("Destroy channel: {} success.", c.first.c_str());
    }

    for (auto& d: aclDatasets) {
        if (acltdtDestroyDataset(d.second) != ACL_ERROR_NONE) {
            LOG_WARN("Destroy dataset: {} failed", d.first.c_str());
        }
    }
    LOG_INFO("Destroy tdt channel end.");
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// 仅用于pybind侧调用，创建tdt cahnnel
void CreateTdtChannel(const string &channelName, int deviceId, int channelSize)
{
    if (!acl_init[deviceId]) {
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        acl_init[deviceId] = true;
    }

    transferChannels[channelName] = tdtCreateChannel(deviceId, channelName.c_str(), channelSize);

    // 创建acltdtDataset类型的数据，对等一个Vector<tensor>。同步接口。
    aclDatasets[channelName] = acltdtCreateDataset();

    LOG_INFO("Create channel {} success", channelName.c_str());
}

bool SendByChannel(const string &channelName, const vector<Tensor> &tensors)
{
    vector<size_t> sizes;
    for (auto& t: tensors) {
        sizes.push_back(t.NumElements());
    }
    string sendName = channelName;

    LOG_DEBUG("Send by {}, count is {}", sendName.c_str(), sizes.size());

    if (sizes.size() == 0) {
        LOG_ERROR("tensors num can not be zero");
        return false;
    }
    bool isNeedResend = false;
    int32_t resendTime = 0;
    tensorflow::Status status = tensorflow::Status::OK();
    do {
        status = tensorflow::SendTensorsByAcl(transferChannels[sendName], ACL_TENSOR_DATA_TENSOR, tensors,
                                              isNeedResend);

        if (status != tensorflow::Status::OK()) {
            LOG_ERROR("Send by {} failed, error {}", sendName.c_str(), status.error_message());
            return false;
        }
        if (resendTime != 0) {
            LOG_ERROR("Send by {} failed, retry: {}", sendName.c_str(), resendTime);
            this_thread::sleep_for(1ms);
        }
        resendTime++;
        if (resendTime > MAX_WAIT_LOOP) {
            LOG_ERROR("Send by {} time out.", sendName.c_str());
            return false;
        }
        LOG_DEBUG("Send by {} success", sendName.c_str());
    } while (isNeedResend);
    return true;
}

void RecvByChannel(const string &channelName, int count)
{
    string recvName = channelName;
    tensorflow::Status status = tensorflow::Status::OK();

    LOG_DEBUG("Transfer try recv: {}", recvName.c_str());
    for (int i = 0; i < count; i++) {
        auto aclStatus = acltdtReceiveTensor(transferChannels[recvName], aclDatasets[recvName], -1);
        if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
            LOG_DEBUG("Transfer recv: {} failed", recvName.c_str());
            continue;
        }

        size_t size = acltdtGetDatasetSize(aclDatasets[recvName]);
        if (size > 0) {
            auto aclData = acltdtGetDataItem(aclDatasets[recvName], i);
            if (aclData != nullptr) {
                char *ptr = reinterpret_cast<char*>(acltdtGetDataAddrFromItem(aclData));
                if (ptr != nullptr) {
                    size_t acl_data_len = acltdtGetDataSizeFromItem(aclData);
                }
            }
        }
    }
    LOG_DEBUG("Transfer recv: {} success", recvName.c_str());
}

// 发送的数据量是value * value * sizeof(float)
int SendByChannelV2(const string &channelName, int count, int value)
{
    tensorflow::DataType dtype = tensorflow::DT_FLOAT;
    tensorflow::TensorShape shape({value, value});
    std::vector<tensorflow::Tensor> tensors;
    size_t dataLen = value * value * sizeof(float);

    // 分配内存空间给张量
    std::default_random_engine e;
    e.seed(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (int item = 0; item < count; ++item) {
        tensorflow::Tensor tensor(dtype, shape);
#if PRECISION_TEST
        float* dataAddr = tensor.flat<float>().data();
#pragma omp parallel for num_threads(OMP_THREADS) default(none) shared(dataLen, dataAddr, distribution, e)
        for (int i = 0; i < (dataLen / sizeof(float)); i++) {
            dataAddr[i] = distribution(e);
        }
#endif
        tensors.emplace_back(std::move(tensor));
    }
    if (SendByChannel(channelName, tensors)) {
        return count;
    } else {
        return 0;
    }
}

void DestroyAclDataset(acltdtDataset *aclDataset, bool includeDataItem)
{
    if (includeDataItem) {
        for (size_t i = 0; i < acltdtGetDatasetSize(aclDataset); i++) {
            if (acltdtDestroyDataItem(acltdtGetDataItem(aclDataset, i)) != ACL_ERROR_NONE) {
                LOG_WARN("Acl destroy tensor data failed.");
            }
        }
    }
    if (acltdtDestroyDataset(aclDataset) != ACL_ERROR_NONE) {
        LOG_WARN("Acl destroy tensor dataset failed.");
    }
}

int SendByAclTdtV2(const string &sendName, float *sendData, int64_t dataLen, int64_t dims[DIM_MAX])
{
    if (sendData == nullptr || dataLen == 0) {
        LOG_ERROR("Send data can not be zero.");
        return 0;
    }

    auto aclDataset = acltdtCreateDataset();
    if (aclDataset == nullptr) {
        LOG_ERROR("Acl create tensor dataset failed");
        return 0;
    }

    acltdtDataItem *aclData =
            acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims, 2, ACL_FLOAT, (char *)(sendData), dataLen);
    if (aclData == nullptr) {
        LOG_ERROR("acltdtCreateDataItem failed");
        DestroyAclDataset(aclDataset, false);
        return 0;
    }

    if (acltdtAddDataItem(aclDataset, aclData) != ACL_ERROR_NONE) {
        LOG_ERROR("acltdtAddDataItem failed");
        acltdtDestroyDataItem(aclData);
        DestroyAclDataset(aclDataset, false);
        return 0;
    }

    LOG_DEBUG("Send by {}, tdt-send start", sendName.c_str());
    auto aclStatus = acltdtSendTensor(transferChannels[sendName], aclDataset, -1);
    DestroyAclDataset(aclDataset, true);
    if (acltdtDestroyDataItem(aclData) != ACL_ERROR_NONE) {
        LOG_WARN("Destroy data item failed.");
    }
    if (aclStatus != ACL_ERROR_NONE) {
        LOG_WARN("Send by {}, tdt-send failed", sendName.c_str());
        return 0;
    }
    LOG_DEBUG("Send by {}, tdt-send success", sendName.c_str());

    return 1;
}

int SendByTdtChannel(const string &channelName, float *sendData, int64_t dims[DIM_MAX])
{
    int64_t dataLen = dims[0] * dims[1] * sizeof(float) * 1L;

    LOG_DEBUG("send by {}, length is {}, dim-0 {}, dim-1 {}",
              channelName.c_str(), dataLen, dims[0], dims[1]);

    return SendByAclTdtV2(channelName, sendData, dataLen, dims);
}
}