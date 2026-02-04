/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void device_timestamp(GM_ADDR output)
{
    // 获取当前系统cycle数
    int64_t systemCycle = AscendC::GetSystemCycle();
   
    // 将结果写入输出缓冲区
    __gm__ int64_t* output_ptr = reinterpret_cast<__gm__ int64_t*>(output);
    *output_ptr = systemCycle;

    // ========== 关键：确保数据写回 GM ==========
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();                          // 1. 等待写指令发射完成
    dcci(output_ptr, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT); // 2. 强制刷 cache 到 GM
    AscendC::PipeBarrier<PIPE_ALL>();                                  // 3. 等待 dcci 完成
}
