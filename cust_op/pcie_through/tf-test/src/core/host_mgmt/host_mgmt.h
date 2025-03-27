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

#ifndef MXREC_HOST_MGMT_H
#define MXREC_HOST_MGMT_H

#include <string>
#include <vector>
#include <thread>

using namespace std;

namespace TfTest {

struct ShmInfo {
    std::string channelName;
    uint64_t memSize;
    int capacity;
    int deviceId;
    std::vector<int64_t> shape;
};

enum class TransferChannel {
    D2H,
    H2D
};

inline string TransferChannel2Str(TransferChannel e)
{
    switch (e) {
        case TransferChannel::D2H:
            return "d2h";
        case TransferChannel::H2D:
            return "h2d";
        default:
            throw std::invalid_argument("Invalid TransferChannel");
    }
};

class HostMgmt {
public:
    HostMgmt() = default;

    ~HostMgmt()
    {
        if (isRunning) {
            Destroy();
        }
    }

    HostMgmt(const HostMgmt&) = delete;

    HostMgmt& operator=(const HostMgmt&) = delete;

    bool Initialize(vector <ShmInfo>& shmInfos, int stepNum, int seed, string type = "RMA");

    void Destroy();

private:
    bool isRunning;
    int stepNum;
    int seed;
    vector <ShmInfo> mgmtShmInfos;
    vector <thread> swapInThreads {};
    vector <thread> swapOutThreads {};
    void *tdtBuffer;
    string type;

    bool CreateChannel(vector <ShmInfo>& shmInfos);

    void RmaSwapTask();

    void TdtSwapTask();

    void JoinRmaSwapThread();
};
}

#endif // MXREC_HOST_MGMT_H