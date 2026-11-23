/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef ASYNCHRONOUS_COMPLETE_CUMSUM_H
#define ASYNCHRONOUS_COMPLETE_CUMSUM_H
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(AsynchronousCompleteCumsumTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalLength); // 处理数据的总长度

    // 分块策略参数
    TILING_DATA_FIELD_DEF(int64_t, totalBlocks);           // 总块数
    TILING_DATA_FIELD_DEF(int64_t, blocksPerCore);         // 每核处理的块数k
    TILING_DATA_FIELD_DEF(int64_t, remainderBlocks);       // 余数块数l
    
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(AsynchronousCompleteCumsum, AsynchronousCompleteCumsumTilingData)
}
#endif // ASYNCHRONOUS_COMPLETE_CUMSUM_H