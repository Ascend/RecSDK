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

#include "kernel_operator.h"
#include "rma_swap_multi_tables.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void rma_swap_multi_tables(GM_ADDR swapInIndex, GM_ADDR swapOutIndex,
                                                            GM_ADDR table_a, GM_ADDR table_b, GM_ADDR table_c,
                                                            GM_ADDR table_d, GM_ADDR table_e, GM_ADDR table_f,
                        GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    int tableNum = tiling_data.tableNum;
    int tableLength = tiling_data.tableLength;
    GM_ADDR svmBuffSwapIn = (GM_ADDR)(tiling_data.shmSwapIn);
    GM_ADDR svmBuffSwapOut = (GM_ADDR)(tiling_data.shmSwapOut);
    uint64_t swapInLen = tiling_data.updateLen;
    int32_t dimNum = tiling_data.dimNum;
    uint64_t dimValue[RMA_SHAPE_DIM_MAX];
    for (int i = 0; i < dimNum; ++i) {
        dimValue[i] = tiling_data.dimValue[i];
    }
    if (TILING_KEY_IS(1)) {
        RmaSwapMultiTables opKernel;
        opKernel.Init(table_a, table_b, table_c, table_d, table_e, table_f,
                      tableNum, tableLength, swapInIndex, swapOutIndex, swapInLen,
                      svmBuffSwapIn, svmBuffSwapOut, usrWorkspace, dimNum, dimValue, output
        );

        opKernel.Process();
    }
}