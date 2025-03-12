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

#include "lcal_sock_exchange.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <securec.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include "asdops/utils/log/log.h"

namespace Lcal {
const std::string LCAL_DEFAULT_SOCK_IP = "127.0.0.1";
constexpr uint16_t LCAL_DEFAULT_SOCK_PORT = 10067;
constexpr uint32_t LCAL_MAX_BACK_LOG = 65535;

int ParseIpAndPort(const char* input, std::string& ip, uint16_t& port)
{
    if (input == nullptr) {
        return LCAL_INVALID_VALUE;
    }
    std::string inputStr(input);
    size_t colonPos = inputStr.find(':');
    if (colonPos == std::string::npos) {
        ASD_LOG(ERROR) << "Input string does not contain a colon separating IP and port.";
        return LCAL_ERROR_INTERNAL;
    }

    ip = inputStr.substr(0, colonPos);
    std::string portStr = inputStr.substr(colonPos + 1);

    std::istringstream portStream(portStr);
    portStream >> port;
    const unsigned short maxPort = 65535;
    if (portStream.fail() || portStream.bad() || port > maxPort) {
        ASD_LOG(ERROR) << "Invalid port number.";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}


LcalSockExchange::~LcalSockExchange()
{
    Cleanup();
}

LcalSockExchange::LcalSockExchange(int rank, int rankSize, std::vector<int>& rankList)
    : rank_(rank), rankSize_(rankSize), rankList_(rankList)
{
}

LcalSockExchange::LcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
    : rank_(rank), rankSize_(rankSize)
{
    lcalCommId_.uid = lcalCommId;
}

int LcalSockExchange::AllGather(const void* sendBuf, size_t sendSize, void* recvBuf)
{
    auto uSendBuf = reinterpret_cast<const uint8_t*>(sendBuf);
    auto uRecvBuf = reinterpret_cast<uint8_t*>(recvBuf);

    if (!isInit_ && Prepare() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }
    isInit_ = true;

    if (!IsServer()) {
        return ClientSendRecv(uSendBuf, sendSize, uRecvBuf);
    }

    return ServerRecvSend(uSendBuf, sendSize, uRecvBuf);
}

int LcalSockExchange::GetNodeNum()
{
    // LCOV_EXCL_START
    if (!isInit_ && Prepare() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }
    isInit_ = true;
    std::ifstream fileStream(UUID_FILE_PATH);
    std::stringstream buffer;
    if (fileStream) {
        buffer << fileStream.rdbuf();
        fileStream.close();
    }
    const std::string uuid = buffer.str();
    ASD_LOG(DEBUG) << "rank:" << rank_ << " UUID " << uuid;

    std::set<std::string> uuidSet {};
    uuidSet.insert(uuid);
    int nodeNum;
    if (IsServer()) {
        for (int i = 1; i < rankSize_; ++i) {
            if (Recv(clientFds_[i], const_cast<__caddr_t>(uuid.data()), uuid.size(), 0) <= 0) {
                ASD_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
            uuidSet.insert(uuid);
        }
        nodeNum = uuidSet.size();
        for (int i = 1; i < rankSize_; ++i) {
            if (Send(clientFds_[i], &nodeNum, sizeof(int), 0) <= 0) {
                ASD_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
        }
    } else {
        if (Send(fd_, uuid.data(), uuid.size(), 0) <= 0) {
            ASD_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
            return LCAL_ERROR_INTERNAL;
        }
        if (Recv(fd_, &nodeNum, sizeof(int), 0) <= 0) {
            ASD_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed ";
            return LCAL_ERROR_INTERNAL;
        }
    }
    return nodeNum;
    // LCOV_EXCL_STOP
}

void LcalSockExchange::GetIpAndPort()
{
    int serverRank = !rankList_.empty() ? rankList_[0] : 0;
    const char* env = getenv("LCAL_COMM_ID");

    if (env == nullptr or ParseIpAndPort(env, ip_, port_) != LCAL_SUCCESS) {
        ip_ = LCAL_DEFAULT_SOCK_IP;
        port_ = LCAL_DEFAULT_SOCK_PORT;
    }
    port_ += serverRank;
    lcalCommId_.handle.addr.sin.sin_family = AF_INET;
    lcalCommId_.handle.addr.sin.sin_addr.s_addr = inet_addr(ip_.c_str());
    lcalCommId_.handle.addr.sin.sin_port = htons(port_);
    ASD_LOG(DEBUG) << "curRank: " << rank_ << " serverRank: " << serverRank << " ip: " << ip_ << " port: " << port_;
}

int LcalSockExchange::Prepare()
{
    if (lcalCommId_.handle.magic != LCAL_MAGIC) {
        GetIpAndPort();
    }
    if (!IsServer()) {
        return Connect();
    }

    clientFds_.resize(rankSize_, -1);
    if (Listen() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }

    if (Accept() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }

    return LCAL_SUCCESS;
}

static bool CheckErrno(int ioErrno)
{
    return ((ioErrno == EAGAIN) || (ioErrno == EWOULDBLOCK) || (ioErrno == EINTR));
}

int LcalSockExchange::Listen()
{
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        ASD_LOG(ERROR) << "Server side create socket failed";
        return LCAL_ERROR_INTERNAL;
    }

    int reuse = 1;
    if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) < 0) {
        ASD_LOG(ERROR) << "Server side set reuseaddr failed";
        return LCAL_ERROR_INTERNAL;
    }

    struct sockaddr *addrPtr = &lcalCommId_.handle.addr.sa;
    if (bind(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
        ASD_LOG(ERROR) << "Server side bind " << lcalCommId_.handle.addr.sin.sin_port << " failed";
        return LCAL_ERROR_INTERNAL;
    }

    /*
     * kernel would silently truncate backlog to the value defined in
     * /proc/sys/net/core/somaxconn if it is less than 65535.
     */
    if (listen(fd_, LCAL_MAX_BACK_LOG) < 0) {
        ASD_LOG(ERROR) << "Server side listen " << lcalCommId_.handle.addr.sin.sin_port << " failed";
        return LCAL_ERROR_INTERNAL;
    }
    ASD_LOG(DEBUG) << "The server is listening! ip: "<< inet_ntoa(lcalCommId_.handle.addr.sin.sin_addr)
        << " port: " << lcalCommId_.handle.addr.sin.sin_port;

    return LCAL_SUCCESS;
}

int LcalSockExchange::AcceptConnection(int fd, sockaddr_in& clientAddr, socklen_t* sinSize)
{
    int clientFd;
    auto* clientAddrPtr = reinterpret_cast<struct sockaddr*>(&clientAddr);

    do {
        clientFd = accept(fd, clientAddrPtr, sinSize);
        if (clientFd < 0) {
            if (!CheckErrno(errno)) {
                ASD_LOG(ERROR) << "Server side accept failed" << strerror(errno);
                return -1;
            }
            ASD_LOG(DEBUG) << "accept failed: " << strerror(errno);
            continue;
        }
        break;
    } while (true);

    return clientFd;
}

int LcalSockExchange::Accept()
{
    struct sockaddr_in clientAddr;
    socklen_t sinSize = sizeof(struct sockaddr_in);

    for (int i = 1; i < rankSize_; ++i) {
        int fd = AcceptConnection(fd_, clientAddr, &sinSize);
        if (fd < 0) {
            ASD_LOG(ERROR) << "AcceptConnection failed";
            return LCAL_ERROR_INTERNAL;
        }

        int rank = 0;
        if (Recv(fd, &rank, sizeof(rank), 0) <= 0) {
            ASD_LOG(ERROR) << "Server side recv rank id failed";
            return LCAL_ERROR_INTERNAL;
        }

        if (rank >= rankSize_ || rank <= 0 || clientFds_[rank] >= 0) {
            ASD_LOG(ERROR) << "Server side recv invalid rank id " << rank;
            return LCAL_ERROR_INTERNAL;
        }

        ASD_LOG(DEBUG) << "Server side recv rank id " << rank;
        clientFds_[rank] = fd;
    }

    return LCAL_SUCCESS;
}

int LcalSockExchange::Send(int fd, const void* sendBuf, size_t sendSize, int flag)
{
    do {
        auto ret = send(fd, sendBuf, sendSize, flag);
        if (ret < 0) {
            if (CheckErrno(errno)) {
                ASD_LOG(ERROR) << "send failed: " << strerror(errno);
                continue;
            }
            ASD_LOG(DEBUG) << "Send failed: " << strerror(errno);
        }
        return ret;
    } while (true);
}

template<typename T>
int LcalSockExchange::Recv(int fd, T* recvBuf, size_t recvSize, int flag)
{
    do {
        auto ret = recv(fd, recvBuf, recvSize, flag);
        if (ret < 0) {
            if (CheckErrno(errno)) {
                ASD_LOG(ERROR) << "recv failed: " << strerror(errno);
                continue;
            }
            ASD_LOG(DEBUG) << "recv failed: " << strerror(errno);
        }
        return ret;
    } while (true);
}

void LcalSockExchange::Close(int &fd)
{
    if (fd == -1) {
        return;
    }

    if (close(fd) < 0) {
        ASD_LOG(WARN) << "failed to close fd:" << fd;
        return;
    }

    fd = -1;
}

int LcalSockExchange::Connect()
{
    ASD_LOG(DEBUG) << "Client side " << rank_ << " begin to connect";

    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        ASD_LOG(ERROR) << "Client side " << rank_ << " create socket failed";
        return LCAL_ERROR_INTERNAL;
    }

    int sleepTimeS = 1;
    int maxRetryCount = 100;
    int retryCount = 0;
    bool success = false;
    struct sockaddr *addrPtr = &lcalCommId_.handle.addr.sa;
    while (retryCount < maxRetryCount) {
        if (connect(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
            if (errno == ECONNREFUSED) {
                ASD_LOG(DEBUG) << "Client side " << rank_ << " try connect " << (retryCount + 1) << " times refused";
                retryCount++;
                sleep(sleepTimeS);
                continue;
            }
            if (errno != EINTR) {
                ASD_LOG(ERROR) << "Client side " << rank_ << " connect failed: " << strerror(errno);
                break;
            }
            ASD_LOG(DEBUG) << "Client side " << rank_ << " try connect failed: " << strerror(errno);
            continue;
        }
        success = true;
        break;
    }

    if (!success) {
        ASD_LOG(ERROR) << "Client side " << rank_ << " connect failed";
        return LCAL_ERROR_INTERNAL;
    }

    if (Send(fd_, &rank_, sizeof(rank_), 0) <= 0) {
        ASD_LOG(ERROR) << "Client side " << rank_ << " send rank failed";
        return LCAL_ERROR_INTERNAL;
    }

    return LCAL_SUCCESS;
}

template<typename T>
int LcalSockExchange::ClientSendRecv(const T* sendBuf, size_t sendSize, T* recvBuf)
{
    if (Send(fd_, sendBuf, sendSize * sizeof(T), 0) <= 0) {
        ASD_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
        return LCAL_ERROR_INTERNAL;
    }

    if (Recv(fd_, recvBuf, sendSize * rankSize_ * sizeof(T), MSG_WAITALL) <= 0) {
        ASD_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed ";
        return LCAL_ERROR_INTERNAL;
    }

    return LCAL_SUCCESS;
}

template<typename T>
int LcalSockExchange::ServerRecvSend(const T* sendBuf, size_t sendSize, T* recvBuf)
{
    auto ret = memcpy_s(recvBuf, sendSize * sizeof (T), sendBuf, sendSize * sizeof (T));
    if (ret != 0) {
        ASD_LOG(ERROR) << "Server side memcpy_s failed, error code:" << std::to_string(ret);
        return LCAL_ERROR_INTERNAL;
    }

    for (int i = 1; i < rankSize_; ++i) {
        if (Recv(clientFds_[i], recvBuf + i * sendSize, sendSize * sizeof(T), MSG_WAITALL) <= 0) {
            ASD_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
            return LCAL_ERROR_INTERNAL;
        }
    }

    for (int i = 1; i < rankSize_; ++i) {
        if (Send(clientFds_[i], recvBuf, sendSize * rankSize_ * sizeof(T), 0) <= 0) {
            ASD_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
            return LCAL_ERROR_INTERNAL;
        }
    }

    return LCAL_SUCCESS;
}

bool LcalSockExchange::IsServer()
{
    return rank_ == 0;
}

void LcalSockExchange::Cleanup()
{
    if (fd_ >= 0) {
        Close(fd_);
    }

    if (clientFds_.empty()) {
        return;
    }

    for (int i = 1; i < rankSize_; ++i) {
        if (clientFds_[i] >= 0) {
            Close(clientFds_[i]);
        }
    }
}

int GetAddrFromString(LcalSocketAddress* ua, const char* ipPortPair)
{
    std::string ip;
    uint16_t port;
    int ret = ParseIpAndPort(ipPortPair, ip, port);
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "lcal ParseIpAndPort failed!";
        return LCAL_ERROR_INTERNAL;
    }
    ua->sin.sin_family = AF_INET;
    ua->sin.sin_addr.s_addr = inet_addr(ip.c_str());
    ua->sin.sin_port = htons(port);
    return LCAL_SUCCESS;
}

int BootstrapGetServerIp(LcalSocketAddress& handle)
{
    char hostname[256];

    // 获取主机名
    if (gethostname(hostname, sizeof(hostname)) < 0) {
        ASD_LOG(ERROR) << "ERROR: Failed to get hostname.";
        return LCAL_ERROR_INTERNAL;
    }

    // 通过主机名获取主机信息
    struct hostent *hostEntry = gethostbyname(hostname);
    if (hostEntry == nullptr) {
        ASD_LOG(ERROR) << "ERROR: Failed to get host entry." ;
        return LCAL_ERROR_INTERNAL;
    }

    // 获取主机的IP地址（仅支持IPv4）
    const char* ip = inet_ntoa(*reinterpret_cast<struct in_addr*>(hostEntry->h_addr_list[0]));
    if (ip == nullptr) {
        ASD_LOG(ERROR) << "ERROR: Failed to convert IP address.";
        return LCAL_ERROR_INTERNAL;
    }

    // 填充 sockaddr_in 结构体
    memset_s(&handle, sizeof(handle), 0, sizeof(handle));
    handle.sin.sin_family = AF_INET;
    handle.sin.sin_addr.s_addr = inet_addr(ip); // 将IP地址填入sockaddr_in
    handle.sin.sin_port = 0;

    return LCAL_SUCCESS;
}
} // namespace Lcal
