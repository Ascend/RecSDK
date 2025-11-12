/**
 * @file relative_attn_bias_time.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */
#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;

namespace {
std::tuple<at::Tensor, at::Tensor> relative_attn_bias_time_impl(const Tensor& timestamps,
                                                                const Tensor& timestampsWeights,
                                                                const double bucketDivisor)
{
    auto timestampsConti = timestamps.contiguous();
    auto timestampsWeightsConti = timestampsWeights.contiguous();
    check_tensor_non_empty(timestampsConti, "timestampsConti");
    check_tensor_non_empty(timestampsWeightsConti, "timestampsWeightsConti");
    check_tensor_dim(timestampsWeightsConti, 2, "timestampsWeightsConti");
    check_tensor_dim(timestampsConti, 2, "timestampsConti");

    // 检查NPU设备且设备ID一致
    std::vector<Tensor> tensors = {timestamps, timestampsWeights};
    std::vector<std::string> names = {"timestamps", "timestampsWeights"};
    check_tensor_npu_device(tensors, names);

    const int64_t numLayers = timestampsWeights.size(0);
    const int64_t bs = timestampsConti.size(0);
    const int64_t s = timestampsConti.size(1);
    constexpr int64_t MAX_S = 4300; // 算子约束最大4300
    TORCH_CHECK(s > 0 && s <= MAX_S, "timestamps size must be between 1 and %d", MAX_S);
    const int64_t sx2 = s * 2;

    at::Tensor rabTimeOut = at::zeros({numLayers, bs, s, 1, s, 1}, timestampsWeightsConti.options());
    at::Tensor bucketTsOut = at::zeros({bs, s, s}, timestampsConti.options());
    EXEC_NPU_CMD(aclnnRelativeAttnBiasTime, timestampsConti, timestampsWeightsConti, bucketDivisor, rabTimeOut,
                 bucketTsOut);
    rabTimeOut = rabTimeOut.repeat({1, 1, 1, 2, 1, 2}).reshape({numLayers, bs, sx2, sx2});
    return {rabTimeOut, bucketTsOut};
}

Tensor relative_attn_bias_time_backward_impl(const Tensor& rabTimeGrad, const Tensor& bucketTimestamps,
                                             const int64_t numBuckets)
{
    check_tensor_non_empty(rabTimeGrad, "rabTimeGrad");
    check_tensor_non_empty(bucketTimestamps, "bucketTimestamps");
    constexpr int CONST_DIM_4 = 4;
    check_tensor_dim(rabTimeGrad, CONST_DIM_4, "rabTimeGrad");

    const int64_t numLayers = rabTimeGrad.size(0);  // rabTimeGrad(n, b, 2s, 2s)
    const int64_t batchsize = rabTimeGrad.size(1);  // rabTimeGrad(n, b, 2s, 2s)
    const int64_t sx2 = rabTimeGrad.size(2);        // rabTimeGrad(n, b, 2s, 2s)
    const int64_t s = sx2 / 2;

    auto rabTimeGradConti = rabTimeGrad.contiguous();
    auto bucketTimestampsConti = bucketTimestamps.contiguous();  // (n, b, s, s)
    bucketTimestampsConti =
        bucketTimestampsConti.view({batchsize, s, 1, s, 1}).repeat({1, 1, 2, 1, 2}).reshape({batchsize, sx2, sx2});

    at::Tensor rabTimeGradOut = at::zeros({numLayers, numBuckets}, rabTimeGrad.options());
    EXEC_NPU_CMD(aclnnRelativeAttnBiasBackward, rabTimeGradConti, bucketTimestampsConti, numBuckets, rabTimeGradOut);
    return rabTimeGradOut;
}
}  // namespace

Tensor relative_attn_bias_time_forward(const Tensor& timestamps, const Tensor& timestampsWeights,
                                       const double bucketDivisor)
{
    auto [rabTimeOut, _] = relative_attn_bias_time_impl(timestamps, timestampsWeights, bucketDivisor);
    return rabTimeOut;
}

tensor_list relative_attn_bias_time_forward_with_index(const Tensor& timestamps, const Tensor& timestampsWeights,
                                                       const double bucketDivisor)
{
    auto [rabTimeOut, bucketTsOut] = relative_attn_bias_time_impl(timestamps, timestampsWeights, bucketDivisor);
    return {rabTimeOut, bucketTsOut};
}

class RelativeAttnBiasTime : public torch::autograd::Function<RelativeAttnBiasTime> {
public:
    static at::Tensor forward(AutogradContext* ctx, const Tensor& timestamps, const Tensor& timestampsWeights,
                              const double bucketDivisor)
    {
        auto [rabTimeOut, bucketTsOut] = relative_attn_bias_time_impl(timestamps, timestampsWeights, bucketDivisor);

        // 保存中间结果供反向使用
        ctx->save_for_backward({bucketTsOut});
        ctx->saved_data["numBuckets"] = timestampsWeights.size(1);
        return rabTimeOut;
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list gradOutputs)
    {
        auto gradOutput = gradOutputs[0];

        auto saved = ctx->get_saved_variables();
        auto bucketTimestamps = saved[0];
        auto numBuckets = ctx->saved_data["numBuckets"].toInt();
        at::Tensor tswGrad = relative_attn_bias_time_backward_impl(gradOutput, bucketTimestamps, numBuckets);
        return {Variable(), tswGrad, Variable()};
    }
};

Tensor relative_attn_bias_time(const Tensor& timestamps, const Tensor& timestampsWeights, const double bucketDivisor)
{
    return RelativeAttnBiasTime::apply(timestamps, timestampsWeights, bucketDivisor);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("relative_attn_bias_time(Tensor timestamps, "
          "                        Tensor timestamps_weights, "
          "                        float bucket_divisor"
          "                        ) -> Tensor");
    m.def("relative_attn_bias_time_backward(Tensor rab_time_grad, "
          "                                 Tensor bucket_timestamps, "
          "                                 int num_buckets"
          "                                 ) -> Tensor");
    m.def("relative_attn_bias_time_with_index(Tensor timestamps, "
          "                                   Tensor timestamps_weights, "
          "                                   float bucket_divisor"
          "                                   ) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("relative_attn_bias_time", &relative_attn_bias_time_forward);
    m.impl("relative_attn_bias_time_backward", &relative_attn_bias_time_backward_impl);
    m.impl("relative_attn_bias_time_with_index", &relative_attn_bias_time_forward_with_index);
}

TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("relative_attn_bias_time", &relative_attn_bias_time);
}
