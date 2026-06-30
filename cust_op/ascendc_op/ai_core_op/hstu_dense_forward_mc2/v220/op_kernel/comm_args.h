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

#ifndef LCCL_COMM_ARGS_H
#define LCCL_COMM_ARGS_H
#include <cstdint>

#if !defined(__DAV_C220_VEC__) && !defined(__DAV_C310__)

#else
#include "kernel_operator.h"
#endif
namespace Lcal {

constexpr int LCAL_MAX_RANK_SIZE = 128;  // Max NPU cards supported by Lcal communication library
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;

constexpr int DFX_COUNT = 50;

constexpr int64_t HALF_NUM = 2;

constexpr int64_t THREE_NUM = 3;

constexpr int64_t FOUR_NUM = 4;

enum Op : int {
    COPYONLY = -1,
    ADD = 0,
    MUL = 1,
    MAX = 2,
    MIN = 3
};

struct ExtraFlag {
    static constexpr uint32_t RDMA = 1;
    static constexpr uint32_t TOPO_910B2C = 1 << 1;
    static constexpr uint32_t TOPO_910_93 = 1 << 2;
    static constexpr uint32_t DETERMINISTIC = 1 << 3;
    static constexpr uint32_t QUANT_FP16 = 1 << 4;
    static constexpr uint32_t QUANT_FP32 = 1 << 5;
    static constexpr uint32_t TOPO_910A5 = 1 << 6;
    static constexpr uint32_t QUANT_DELAY = 1 << 7;
    static constexpr uint32_t QUANT_CURRENT = 1 << 8;
    static constexpr uint32_t TOPO_PCIE = 1 << 9;
    static constexpr uint32_t ATOMIC_ENABLE = 1 << 15;  // Enable atomic ops on 910A5
};

struct CommArgs {
    int rank = 0;  // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0;                           // global rank size
    int localRankSize = -1;                     // Number of cards connected via fullmesh
    uint32_t extraFlag = 0;                     // 32-bit bitmap; each bit's meaning is defined above in ExtraFlag
    GM_ADDR peerMems[LCAL_MAX_RANK_SIZE] = {};  // Shared memory buffers from initialization; same for all AllReduce ops
    /**
     * @param sendCountMatrix 1-D array of size rankSize*rankSize.
     * e.g., sendCountMatrix[1] maps to 2-D coordinate [0][1], meaning number of elements
     *       that card 0 sends to card 1.
     */
    int64_t sendCountMatrix[LCAL_MAX_RANK_SIZE * LCAL_MAX_RANK_SIZE] = {};  // for all2allv
    int64_t dfx[DFX_COUNT] = {};
};
}  // namespace Lcal
#endif  // LCCL_COMM_ARGS_H
