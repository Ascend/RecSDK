/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#include "disetangle_attention.h"

template <typename T>
__aicore__ inline void InvokeImpl(const KernelArgs &kernel_args)
{
    TPipe t_pipe;
    T kernel;
    GET_TILING_DATA(tiling_data, kernel_args.tiling);
    REGIST_MATMUL_OBJ(&t_pipe,
        GetSysWorkSpacePtr(),
        kernel.P2C_MM,
        &tiling_data.MM,
        kernel.C2P_MM,
        &tiling_data.MM,
        kernel.QK_MM,
        &tiling_data.QK_MM,
        kernel.SV_MM,
        &tiling_data.SV_MM);

    kernel.init(kernel_args, tiling_data, &t_pipe);
    kernel.process();
}

extern "C" __global__ __aicore__ void disetangle_attention(GM_ADDR query_layer, GM_ADDR key_layer, GM_ADDR value_layer,
    GM_ADDR pos_key_layer, GM_ADDR pos_query_layer, GM_ADDR relative_pos, GM_ADDR atten_mask,
    GM_ADDR atten_outputs,  // output 0
    GM_ADDR atten_probs,    // output 1
    GM_ADDR atten_weights,  // output 2
    GM_ADDR workspace, GM_ADDR tiling)
{
    KernelArgs kernel_args = {query_layer,
        key_layer,
        value_layer,
        pos_key_layer,
        pos_query_layer,
        relative_pos,
        atten_mask,
        atten_outputs,
        atten_probs,
        atten_weights,
        workspace,
        tiling};

    if (TILING_KEY_IS(0)) {  // only c2p mode
        InvokeImpl<DisetangleAttention<DTYPE_QUERY_LAYER, true, false>>(kernel_args);
    } else if (TILING_KEY_IS(1)) {  // only p2c mode
        InvokeImpl<DisetangleAttention<DTYPE_QUERY_LAYER, false, true>>(kernel_args);
    } else if (TILING_KEY_IS(2)) {  // c2p + p2c mode
        InvokeImpl<DisetangleAttention<DTYPE_QUERY_LAYER, true, true>>(kernel_args);
    }
}