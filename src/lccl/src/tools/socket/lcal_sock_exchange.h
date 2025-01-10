/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef LCCL_SOCK_EXCHANGE_H
#define LCCL_SOCK_EXCHANGE_H

#include <vector>
#include <string>
#include <memory>
#include <ctime>

#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>

#include "lcal_types.h"
#include "lcal_api.h"

namespace Lcal {
/* Common socket address storage structure for IPv4/IPv6 */
union LcalSocketAddress {
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

const std::string UUID_FILE_PATH = "/proc/sys/kernel/random/boot_id";
// mask for setting ip and port
constexpr uint64_t LCAL_MAGIC = 0xdddd0000dddd0000;

struct LcalBootstrapHandle {
    uint64_t magic;
    union LcalSocketAddress addr;
};

union LcalBootstrap {
    LcalBootstrapHandle handle;
    LcalUniqueId uid;
};

class LcalSockExchange {
public:
    LcalSockExchange(int rank, int rankSize, std::vector<int>& rankList);
    LcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId);
    ~LcalSockExchange();

    /* *
     * @brief All gather data from @ref sendBuf to @ref recvBuf
     *
     * @note recvBuf's space must larger than sendSize * rankSize_
     * @return LCAL_SUCCESS for success, other for failed
     */
    int AllGather(const void* sendBuf, size_t sendSize, void* recvBuf);

    int GetNodeNum();

    static bool CheckValid(LcalUniqueId lcalCommId)
    {
        LcalBootstrap id {};
        id.uid = lcalCommId;
        return id.handle.magic == LCAL_MAGIC;
    }

private:
    void GetIpAndPort();
    int Prepare();
    int Listen();
    int Accept();
    int Send(int fd, const void* sendBuf, size_t sendSize, int flag);
    template <typename T> int Recv(int fd, T* recvBuf, size_t recvSize, int flag);
    void Close(int& fd);
    int Connect();
    int AcceptConnection(int fd, sockaddr_in& clientAddr, socklen_t* sinSize);

    template <typename T> int ClientSendRecv(const T* sendBuf, size_t sendSize, T* recvBuf);
    template <typename T> int ServerRecvSend(const T* sendBuf, size_t sendSize, T* recvBuf);
    void Cleanup();

    bool IsServer();

    int rank_ = 0;
    int rankSize_ = 0;
    int fd_ = -1;
    std::vector<int> clientFds_;
    bool isInit_ = false;
    std::vector<int> rankList_ = {};
    std::string ip_;
    uint16_t port_ = 0;
    LcalBootstrap lcalCommId_ = {};
};
} // namespace Lcal

#endif