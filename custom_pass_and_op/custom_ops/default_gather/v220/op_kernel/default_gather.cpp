/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
using namespace AscendC;
#define BUFFER_NUM 1
template <typename T>
class CustomGather {
public:
    __aicore__ inline CustomGather() {}

    __aicore__ inline void init(GM_ADDR id, GM_ADDR table, GM_ADDR output, GM_ADDR workspace)
    {
        pipe.InitBuffer(input_table, 1, table_row_copyin * sizeof(DTYPE_TABLE));
        pipe.InitBuffer(input_id, 1, id_column_copyin * sizeof(DTYPE_ID));
        pipe.InitBuffer(output_table, 1, table_row * sizeof(DTYPE_TABLE));
        aGM.SetGlobalBuffer((__gm__ DTYPE_ID*)id, id_column);                       // id
        bGM.SetGlobalBuffer((__gm__ DTYPE_TABLE*)table, table_column * table_row);  // table
        dstDataGm.SetGlobalBuffer((__gm__ DTYPE_TABLE*)output, id_column * table_row);
    }
    __aicore__ inline void init_para(GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingdata, tiling);
        Block_num = block_num;
        block_id = GetBlockIdx();
        id_column = tilingdata.Idcolumn;
        id_column_copyin = tilingdata.IdColumnCopyin;
        table_column = tilingdata.Tablecolumn;
        table_row = tilingdata.Tablerow;
        table_row_copyin = tilingdata.TableRowCopyin;
        // compute part means how many row
        min_part_num = tilingdata.MinPartNum;
        if (block_id < min_part_num) {
            single_core_compute_part = tilingdata.MinSingleCoreComputePart;
            index_start = single_core_compute_part * block_id;
        } else {
            single_core_compute_part = tilingdata.MaxSingleCoreComputePart;
            index_start = single_core_compute_part * block_id - min_part_num;
        }
        loopcount = single_core_compute_part;
    }
    __aicore__ inline void process()
    {
        // copypad relay..
        DataCopyExtParams dataCopyParams{1, table_row * (uint32_t)sizeof(DTYPE_TABLE), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_TABLE> dataPadParams{true, 0, 0, 0};

        // 最外层只alloc ID ，因为我们的思路是一次性搬完，这样就不需要考虑padding
        LocalTensor<DTYPE_ID> idLocal = input_id.AllocTensor<DTYPE_ID>();
        DataCopy(idLocal, aGM[index_start], id_column_copyin);
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < loopcount; i++) {
            // 第二层 alloc Output Table  单行搬出 normal 版本
            LocalTensor<DTYPE_TABLE> outLocal = output_table.AllocTensor<T>();
            int lookup_idx = idLocal.GetValue(i);
            if (lookup_idx >= 0) {
                LocalTensor<DTYPE_TABLE> tableLocal = input_table.AllocTensor<DTYPE_TABLE>();
                DataCopy(tableLocal, bGM[lookup_idx * table_row], table_row_copyin);
                input_table.EnQue(tableLocal);
                tableLocal = input_table.DeQue<DTYPE_TABLE>();
                Adds(outLocal, tableLocal, (DTYPE_TABLE)(0), table_row_copyin);
                input_table.FreeTensor(tableLocal);  // 释放 单行的 table
            } else {
                Duplicate(outLocal, (DTYPE_TABLE)(0), table_row_copyin);
            }
            output_table.EnQue(outLocal);
            outLocal = output_table.DeQue<DTYPE_TABLE>();

            DataCopyPad(dstDataGm[index_start * table_row + i * table_row], outLocal, dataCopyParams);  // + pad
            output_table.FreeTensor(outLocal);  // 释放多行的out table
        }
        input_id.FreeTensor(idLocal);  // 唯一一次释放
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> input_table;
    TQue<TPosition::VECIN, 1> input_id;
    TQue<TPosition::VECOUT, 1> output_table;
    GlobalTensor<DTYPE_ID> aGM;
    GlobalTensor<DTYPE_TABLE> bGM, dstDataGm;
    int32_t Block_num, block_id, ubsize, id_column, table_column, table_row, single_core_compute_part, loopcount,
        table_row_copyin, id_column_copyin, min_part_num, index_start;
};
template <typename T>
/// datacopy + 全载
class CustomGathera {
public:
    __aicore__ inline CustomGathera() {}

    __aicore__ inline void init(GM_ADDR id, GM_ADDR table, GM_ADDR output, GM_ADDR workspace)
    {
        pipe.InitBuffer(input_table, 1, table_row_copyin * table_column * sizeof(DTYPE_TABLE));
        pipe.InitBuffer(input_id, 1, id_column_copyin * sizeof(DTYPE_ID));
        pipe.InitBuffer(output_table, 1, single_core_compute_part * table_row * sizeof(DTYPE_TABLE));
        aGM.SetGlobalBuffer((__gm__ DTYPE_ID*)id, id_column);                       // id
        bGM.SetGlobalBuffer((__gm__ DTYPE_TABLE*)table, table_column * table_row);  // table
        dstDataGm.SetGlobalBuffer((__gm__ DTYPE_TABLE*)output, id_column * table_row);
    }
    __aicore__ inline void init_para(GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingdata, tiling);
        Block_num = block_num;
        block_id = GetBlockIdx();
        id_column = tilingdata.Idcolumn;
        id_column_copyin = tilingdata.IdColumnCopyin;
        table_column = tilingdata.Tablecolumn;
        table_row = tilingdata.Tablerow;
        table_row_copyin = tilingdata.TableRowCopyin;

        min_part_num = tilingdata.MinPartNum;
        if (block_id < min_part_num) {
            single_core_compute_part = tilingdata.MinSingleCoreComputePart;
            index_start = single_core_compute_part * block_id;
        } else {
            single_core_compute_part = tilingdata.MaxSingleCoreComputePart;
            index_start = single_core_compute_part * block_id - min_part_num;
        }
        loopcount = single_core_compute_part;
    }
    __aicore__ inline void process()
    {
        // 最外层只alloc ID ，因为我们的思路是一次性搬完，这样就不需要考虑padding
        LocalTensor<DTYPE_ID> idLocal = input_id.AllocTensor<DTYPE_ID>();
        LocalTensor<DTYPE_TABLE> tableLocal = input_table.AllocTensor<DTYPE_TABLE>();
        LocalTensor<DTYPE_TABLE> outLocal = output_table.AllocTensor<DTYPE_TABLE>();
        DataCopy(tableLocal, bGM, table_row_copyin * table_column);  // todo 改搬运长度32b对齐
        DataCopy(idLocal, aGM[index_start], id_column_copyin);
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < loopcount; i++) {
            int lookup_idx = idLocal.GetValue(i);
            if (lookup_idx >= 0) {
                Adds(outLocal[i * table_row_copyin], tableLocal[lookup_idx * table_row_copyin], (DTYPE_TABLE)(0),
                     table_row_copyin);
            } else {
                Duplicate(outLocal[i * table_row_copyin], (DTYPE_TABLE)(0), table_row_copyin);
            }
        }
        output_table.EnQue(outLocal);
        outLocal = output_table.DeQue<DTYPE_TABLE>();
        DataCopy(dstDataGm[index_start * table_row], outLocal, single_core_compute_part * table_row);  // + pad
        input_id.FreeTensor(idLocal);                                                                  // 唯一一次释放
        input_table.FreeTensor(tableLocal);  // 释放 单行的 table
        output_table.FreeTensor(outLocal);   // 释放多行的out table
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> input_table;
    TQue<TPosition::VECIN, 1> input_id;
    TQue<TPosition::VECOUT, 1> output_table;
    GlobalTensor<DTYPE_ID> aGM;
    GlobalTensor<DTYPE_TABLE> bGM, dstDataGm;
    int32_t Block_num, block_id, ubsize, id_column, table_column, table_row, single_core_compute_part, loopcount,
        table_row_copyin, id_column_copyin, min_part_num, index_start;
};

template <typename T>
// datacopy + 单行搬 + 1/2 ub table搬运
class CustomGatherb {
public:
    __aicore__ inline CustomGatherb() {}

    __aicore__ inline void init(GM_ADDR id, GM_ADDR table, GM_ADDR output, GM_ADDR workspace)
    {
        pipe.InitBuffer(input_table_total, 1, row_in_ub * table_row_copyin * sizeof(DTYPE_TABLE));
        pipe.InitBuffer(input_table, 1, table_row_copyin * sizeof(DTYPE_TABLE));
        pipe.InitBuffer(input_id, 1, id_column_copyin * sizeof(DTYPE_ID));
        pipe.InitBuffer(output_table, 1, row_in_ub * table_row * sizeof(DTYPE_TABLE));
        aGM.SetGlobalBuffer((__gm__ DTYPE_ID*)id, id_column);                       // id
        bGM.SetGlobalBuffer((__gm__ DTYPE_TABLE*)table, table_column * table_row);  // table
        dstDataGm.SetGlobalBuffer((__gm__ DTYPE_TABLE*)output, id_column * table_row);
    }
    __aicore__ inline void init_para(GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingdata, tiling);
        Block_num = block_num;
        block_id = GetBlockIdx();
        id_column = tilingdata.Idcolumn;
        id_column_copyin = tilingdata.IdColumnCopyin;
        table_column = tilingdata.Tablecolumn;
        table_row = tilingdata.Tablerow;
        table_row_copyin = tilingdata.TableRowCopyin;
        row_in_ub = tilingdata.RowInUb;

        min_part_num = tilingdata.MinPartNum;
        if (block_id < min_part_num) {
            single_core_compute_part = tilingdata.MinSingleCoreComputePart;
            index_start = single_core_compute_part * block_id;
            times_copyout = tilingdata.MinTimesCopyOut;
        } else {
            single_core_compute_part = tilingdata.MaxSingleCoreComputePart;
            index_start = single_core_compute_part * block_id - min_part_num;
            times_copyout = tilingdata.MaxTimesCopyOut;
        }
        loopcount = single_core_compute_part;
        last_time_copyout = single_core_compute_part % row_in_ub;
    }
    __aicore__ inline void process()
    {
        // 最外层只alloc ID ，因为我们的思路是一次性搬完，这样就不需要考虑padding
        LocalTensor<DTYPE_ID> idLocal = input_id.AllocTensor<DTYPE_ID>();
        LocalTensor<DTYPE_TABLE> tableLocalTotal = input_table_total.AllocTensor<DTYPE_TABLE>();
        LocalTensor<DTYPE_TABLE> outLocal = output_table.AllocTensor<DTYPE_TABLE>();
        DataCopy(tableLocalTotal, bGM, row_in_ub * table_row_copyin);
        DataCopy(idLocal, aGM[index_start], id_column_copyin);
        PipeBarrier<PIPE_ALL>();
        for (int i = 0; i < loopcount; i++) {
            // 第二层 alloc Output Table  单行搬出 normal 版本
            int lookup_idx = idLocal.GetValue(i);
            if (lookup_idx >= row_in_ub) {
                DataCopy(outLocal[(i % row_in_ub) * table_row], bGM[lookup_idx * table_row_copyin], table_row_copyin);
            } else if (lookup_idx >= 0) {
                Adds(outLocal[(i % row_in_ub) * table_row], tableLocalTotal[lookup_idx * table_row], (DTYPE_TABLE)(0),
                     table_row);
            } else if (lookup_idx < 0) {
                Duplicate(outLocal[(i % row_in_ub) * table_row], (DTYPE_TABLE)(0), table_row);
            }

            if (i % row_in_ub == row_in_ub - 1) {
                output_table.EnQue(outLocal);
                outLocal = output_table.DeQue<DTYPE_TABLE>();
                PipeBarrier<PIPE_ALL>();
                DataCopy(dstDataGm[index_start * table_row + (i / row_in_ub) * row_in_ub * table_row_copyin], outLocal,
                         row_in_ub * table_row);  // + pad
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        output_table.EnQue(outLocal);
        outLocal = output_table.DeQue<DTYPE_TABLE>();
        DataCopy(dstDataGm[index_start * table_row + (single_core_compute_part - last_time_copyout) * table_row],
                 outLocal, last_time_copyout * table_row);
        output_table.FreeTensor(outLocal);  // 释放多行的out table
        input_table_total.FreeTensor(tableLocalTotal);
        input_id.FreeTensor(idLocal);  // 唯一一次释放
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> input_table;
    TQue<TPosition::VECIN, 1> input_id;
    TQue<TPosition::VECIN, 1> input_table_total;
    TQue<TPosition::VECOUT, 1> output_table;
    GlobalTensor<DTYPE_ID> aGM;
    GlobalTensor<DTYPE_TABLE> bGM, dstDataGm;
    int32_t Block_num, block_id, ubsize, id_column, table_column, table_row, single_core_compute_part, loopcount,
        table_row_copyin, id_column_copyin, last_time_copyout, row_in_ub, times_copyout, min_part_num, index_start;
};

extern "C" __global__ __aicore__ void default_gather(GM_ADDR id, GM_ADDR table, GM_ADDR output, GM_ADDR workspace,
                                                     GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        CustomGathera<DTYPE_TABLE> opp;
        opp.init_para(tiling);
        opp.init(id, table, output, workspace);
        opp.process();
    } else if (TILING_KEY_IS(2)) {
        CustomGatherb<DTYPE_TABLE> opp;
        opp.init_para(tiling);
        opp.init(id, table, output, workspace);
        opp.process();

    } else if (TILING_KEY_IS(3)) {
        CustomGather<DTYPE_TABLE> opp;
        opp.init_para(tiling);
        opp.init(id, table, output, workspace);
        opp.process();
    }
}
