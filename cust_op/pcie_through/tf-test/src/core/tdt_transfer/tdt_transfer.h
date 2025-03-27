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

#ifndef MXREC_TDT_TRANSFER_H
#define MXREC_TDT_TRANSFER_H

#include "acl/acl_base.h"
#include "acl/acl.h"
#include "acl/acl_tdt.h"
#include "acl/acl_tdt_queue.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;

constexpr int32_t DIM_MAX = 2;

namespace TfTest {
void FreeTdtChannel(int deviceId);
void CreateTdtChannel(const string &channelName, int deviceId, int channelSize);
bool SendByChannel(const string &channelName, const vector<Tensor> &tensors);
void RecvByChannel(const string &channelName, int count);
int SendByChannelV2(const string &channelName, int count, int value);
void DestroyAclDataset(acltdtDataset *aclDataset, bool includeDataItem);
int SendByAclTdtV2(const string &sendName, float *sendData, int64_t dataLen, int64_t dims[DIM_MAX]);
int SendByTdtChannel(const string &channelName, float *sendData, int64_t dims[DIM_MAX]);
}

#endif // MXREC_TDT_TRANSFER_H