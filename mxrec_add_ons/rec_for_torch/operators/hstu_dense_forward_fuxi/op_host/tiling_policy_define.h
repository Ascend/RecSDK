#ifndef TILING_POLICY_DEFINE_H
#define TILING_POLICY_DEFINE_H

namespace HstuDenseForwardFuxi {

#include <cstdio>
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace INDEX_T {
    constexpr int INDEX_0 = 0;
    constexpr int INDEX_1 = 1;
    constexpr int INDEX_2 = 2;
    constexpr int INDEX_3 = 3;
    constexpr int INDEX_4 = 4;
    constexpr int INDEX_5 = 5;
}

constexpr int FLOAT_TILING_KEY = 2;
constexpr int BF16_TILING_KEY = 1;
constexpr int FLOAT16_TILING_KEY = 0;

constexpr int MAX_AIV_NUM = 48;
constexpr int MAX_BATCH_SIZE = 10;
constexpr int BLOCK_HEIGHT = 64;
constexpr int VCORE_NUM_IN_ONE_AIC = 1;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;
constexpr int TRANS_TASK_NUM = 3;

constexpr int OUTPUT_DIM_NUM = 3;

}

#endif