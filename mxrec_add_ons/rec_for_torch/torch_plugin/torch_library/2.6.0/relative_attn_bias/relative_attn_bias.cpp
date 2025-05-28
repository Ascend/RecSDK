/**
 * @file relative_attn_bias.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using namespace at;
using namespace std;

std::tuple<Tensor, Tensor> relative_attn_bias_impl_npu(const Tensor& relPosBias, const Tensor& identity,
                                                       const Tensor& timestamps, const Tensor& timestampsWeights,
                                                       const at::IntArrayRef pastValidLens, const double bucketDivisor)
{
    auto relPosBiasConti = relPosBias.contiguous();
    auto identityConti = identity.contiguous();
    auto timestampsConti = timestamps.contiguous();
    auto timestampsWeightsConti = timestampsWeights.contiguous();

    const int bs = pastValidLens.size();
    const int sx2 = relPosBias.size(0);  // relPosBias(2s, 2s)
    const int s = sx2 / 2;
    const int numLayers = timestampsWeights.size(0);

    at::Tensor rabPosOut = at::zeros({bs, sx2, sx2}, relPosBiasConti.options());
    at::Tensor rabTimeOut = at::zeros({numLayers, bs, s, 1, s, 1}, timestampsWeightsConti.options());

    EXEC_NPU_CMD(aclnnRelativeAttnBias, relPosBiasConti, identityConti, timestampsConti, timestampsWeightsConti,
                 pastValidLens, bucketDivisor, rabPosOut, rabTimeOut);
    rabTimeOut = rabTimeOut.repeat({1, 1, 1, 2, 1, 2}).reshape({numLayers, bs, sx2, sx2});
    return {rabPosOut, rabTimeOut};
}

Tensor relative_attn_bias_backward_impl_npu(const Tensor& rabTimeGrad, const Tensor& bucketTimestamps,
                                            const int64_t numBuckets)
{
    const int numLayers = rabTimeGrad.size(0);  // rabTimeGrad(n, b, 2s, 2s)
    const int batchsize = rabTimeGrad.size(1);  // rabTimeGrad(n, b, 2s, 2s)
    const int sx2 = rabTimeGrad.size(2);        // rabTimeGrad(n, b, 2s, 2s)
    const int s = sx2 / 2;

    auto rabTimeGradConti = rabTimeGrad.contiguous();
    auto bucketTimestampsConti = bucketTimestamps.contiguous();  // (n, b, s, s)
    bucketTimestampsConti = bucketTimestampsConti.view({batchsize, s, 1, s, 1})
                                                 .repeat({1, 1, 2, 1, 2})
                                                 .reshape({batchsize, sx2, sx2});

    at::Tensor rabTimeGradOut = at::zeros({numLayers, numBuckets}, rabTimeGrad.options());
    EXEC_NPU_CMD(aclnnRelativeAttnBiasBackward, rabTimeGradConti, bucketTimestampsConti, numBuckets, rabTimeGradOut);
    return rabTimeGradOut;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("relative_attn_bias(Tensor rel_pos_bias, "
          "                   Tensor identity, "
          "                   Tensor timestamps, "
          "                   Tensor timestamps_weights, "
          "                   int[] past_valid_lens,"
          "                   float bucket_divisor"
          "                   ) -> (Tensor, Tensor)");
    m.def("relative_attn_bias_backward(Tensor rab_time_grad, "
          "                            Tensor bucket_timestamps, "
          "                            int num_buckets"
          "                            ) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("relative_attn_bias", &relative_attn_bias_impl_npu);
    m.impl("relative_attn_bias_backward", &relative_attn_bias_backward_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("relative_attn_bias", &relative_attn_bias_impl_npu);
    m.impl("relative_attn_bias_backward", &relative_attn_bias_backward_impl_npu);
}
