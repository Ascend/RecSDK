/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


constexpr int32_t MAX_WEIGHTS_OFFSETS_SIZE = 1024;

void validate_common_inputs(
    const at::Tensor &devWeights,
    const at::Tensor &weightsOffsets,
    const at::Tensor &dOffsets,
    const at::Tensor &hashSizeCumsum,
    const at::Tensor &indices,
    const at::Tensor &offsets)
{
    check_tensor_non_empty(devWeights, "devWeights");
    check_tensor_non_empty(weightsOffsets, "weightsOffsets");
    check_tensor_non_empty(dOffsets, "dOffsets");
    check_tensor_non_empty(hashSizeCumsum, "hashSizeCumsum");
    check_tensor_non_empty(indices, "indices");
    check_tensor_non_empty(offsets, "offsets");

    check_tensor_dim(devWeights, 1, "devWeights");
    check_tensor_dim(weightsOffsets, 1, "weightsOffsets");
    check_tensor_dim(dOffsets, 1, "dOffsets");
    check_tensor_dim(hashSizeCumsum, 1, "hashSizeCumsum");
    check_tensor_dim(indices, 1, "indices");
    check_tensor_dim(offsets, 1, "offsets");

    TORCH_CHECK(devWeights.dtype() == at::kFloat, "devWeights must be float type");
    TORCH_CHECK(indices.dtype() == at::kInt || indices.dtype() == at::kLong,
        "indices must be int or long type");

    TORCH_CHECK(weightsOffsets.dtype() == at::kLong || weightsOffsets.dtype() == at::kInt,
        "weightsOffsets must be int or long type");
    TORCH_CHECK(weightsOffsets.size(0) <= MAX_WEIGHTS_OFFSETS_SIZE,
        "weightsOffsets size must be less than or equal to 1024 to prevent array overflow");

    TORCH_CHECK(offsets.dtype() == at::kInt || offsets.dtype() == at::kLong,
        "offsets must be int or long type");
    TORCH_CHECK(offsets.size(0) > 0, "offsets size must be greater than 0");
    
    TORCH_CHECK(dOffsets.dtype() == at::kLong || dOffsets.dtype() == at::kInt,
        "dOffsets must be int or long type");
    TORCH_CHECK(hashSizeCumsum.dtype() == at::kLong || hashSizeCumsum.dtype() == at::kInt,
        "hashSizeCumsum must be int or long type");

    check_tensor_npu_device(
        {devWeights, weightsOffsets, dOffsets, hashSizeCumsum, indices, offsets},
        {"devWeights", "weightsOffsets", "dOffsets", "hashSizeCumsum", "indices", "offsets"});
}

void validate_dense_embedding_codegen_lookup_function_inputs(
    const at::Tensor &devWeights,
    const at::Tensor &weightsOffsets,
    const at::Tensor &dOffsets,
    const at::Tensor &hashSizeCumsum,
    const at::Tensor &indices,
    const at::Tensor &offsets,
    const c10::optional<at::Tensor> &indiceWeights,
    const c10::optional<at::Tensor> &featureRequiresGrad)
{
    validate_common_inputs(devWeights, weightsOffsets, dOffsets, hashSizeCumsum, indices, offsets);

    TORCH_CHECK(devWeights.dtype() == at::kFloat, "devWeights must be float type");
    if (indiceWeights.has_value()) {
        TORCH_CHECK(indiceWeights.value().dtype() == at::kFloat, "indiceWeights must be float type");
    }

    if (featureRequiresGrad.has_value()) {
        TORCH_CHECK(featureRequiresGrad.value().dtype() == at::kByte, "featureRequiresGrad must be byte type");
    }

    auto offsets_last = offsets[-1].item<int64_t>();
    TORCH_CHECK(offsets_last == indices.size(0),
        "offsets last element must match indices size, but got ", offsets_last, " vs ", indices.size(0));
}

// 校验dense_embedding_codegen_lookup_function_grad算子的输入参数
void validate_dense_embedding_codegen_lookup_function_grad_inputs(
    const at::Tensor &devWeights,
    const at::Tensor &weightsGrad,
    const at::Tensor &weightsOffsets,
    const at::Tensor &dOffsets,
    const at::Tensor &hashSizeCumsum,
    const at::Tensor &indices,
    const at::Tensor &offsets)
{
    validate_common_inputs(devWeights, weightsOffsets, dOffsets, hashSizeCumsum, indices, offsets);

    check_tensor_non_empty(weightsGrad, "weightsGrad");
    check_tensor_dim(weightsGrad, 2, "weightsGrad");
    TORCH_CHECK(weightsGrad.dtype() == at::kFloat, "weightsGrad must be float type");

    TORCH_CHECK(weightsGrad.size(0) > 0, "weightsGrad size must be greater than 0");
    check_tensor_npu_device(
        {weightsGrad, devWeights, weightsOffsets, dOffsets, hashSizeCumsum, indices, offsets},
        {"weightsGrad", "devWeights", "weightsOffsets", "dOffsets", "hashSizeCumsum", "indices", "offsets"});

    auto offsets_last = offsets[-1].item<int64_t>();
    TORCH_CHECK(offsets_last == indices.size(0),
        "offsets last element must match indices size, but got ", offsets_last, " vs ", indices.size(0));
}

// dense_embedding_codegen_lookup_function算子的前向实现
at::Tensor dense_embedding_codegen_lookup_function_impl_npu(
    const at::Tensor &devWeights,
    const at::Tensor &weightsOffsets,
    const at::Tensor &dOffsets,
    const int64_t totalD,
    const int64_t maxD,
    const at::Tensor &hashSizeCumsum,
    const int64_t totalHashSizeBits,
    const at::Tensor &indices,
    const at::Tensor &offsets,
    const int64_t poolingMode,
    const std::optional<at::Tensor> &indiceWeightsOptional,
    const std::optional<at::Tensor> &featureRequiresGrad,
    const int64_t outputDtypeOptional,
    const std::optional<at::Tensor> &bOffsetOptional,
    const std::optional<at::Tensor> &vbeOutputOffsetsFeatureRankOptional,
    const std::optional<at::Tensor> &vbeBOffsetsRankPerFeatureOptional,
    const int64_t maxB,
    const int64_t maxBFeatureRank,
    const int64_t vbeOutputSize)
{
    validate_dense_embedding_codegen_lookup_function_inputs(
        devWeights, weightsOffsets, dOffsets, hashSizeCumsum,
        indices, offsets, indiceWeightsOptional, featureRequiresGrad);

    auto devWeightsCon = devWeights.contiguous();
    auto weightsOffsetsCon = weightsOffsets.contiguous();
    auto dOffsetsCon = dOffsets.contiguous().to(at::kLong);
    auto hashSizeCumsumCon = hashSizeCumsum.contiguous();
    auto indicesCon = indices.contiguous();
    auto offsetsCon = offsets.contiguous();
    auto indiceWeightsOptionalCon = indiceWeightsOptional.value_or(Tensor()).contiguous();
    auto bOffsetOptionalCon = bOffsetOptional.value_or(Tensor()).contiguous();
    auto vbeOutputOffsetsFeatureRankOptionalCon = vbeOutputOffsetsFeatureRankOptional.value_or(Tensor()).contiguous();
    auto vbeBOffsetsRankPerFeatureOptionalCon = vbeBOffsetsRankPerFeatureOptional.value_or(Tensor()).contiguous();
    at::Tensor out = at::empty({indicesCon.size(0), maxD}, devWeightsCon.options());
    const bool enableOptimization = true;
    EXEC_NPU_CMD(aclnnDenseEmbeddingCodegenLookupFunction,
                 devWeightsCon, weightsOffsetsCon, dOffsetsCon,
                 hashSizeCumsumCon, indicesCon, offsetsCon, indiceWeightsOptionalCon, bOffsetOptionalCon,
                 vbeOutputOffsetsFeatureRankOptionalCon, vbeBOffsetsRankPerFeatureOptionalCon,
                 totalD, maxD, totalHashSizeBits,
                 poolingMode, enableOptimization, outputDtypeOptional, maxB, maxBFeatureRank, vbeOutputSize, out);

    return out;
}

// 为NPU设备注册反向实现
std::vector<at::Tensor> dense_embedding_codegen_lookup_function_grad_impl_npu(
    const at::Tensor &devWeights,
    const at::Tensor &weightsGrad,
    const at::Tensor &weightsOffsets,
    const at::Tensor &dOffsets,
    const int64_t totalD,
    const int64_t maxD,
    const at::Tensor &hashSizeCumsum,
    const int64_t totalHashSizeBits,
    const at::Tensor &indices,
    const at::Tensor &offsets,
    const int64_t poolingMode,
    const std::optional<at::Tensor> &indiceWeightsOptional,
    const std::optional<at::Tensor> &featureRequiresGrad,
    const int64_t outputDtypeOptional,
    const std::optional<at::Tensor> &bOffsetOptional,
    const std::optional<at::Tensor> &vbeOutputOffsetsFeatureRankOptional,
    const std::optional<at::Tensor> &vbeBOffsetsRankPerFeatureOptional,
    const int64_t maxB,
    const int64_t maxBFeatureRank,
    const int64_t vbeOutputSize)
{
    validate_dense_embedding_codegen_lookup_function_grad_inputs(
        devWeights, weightsGrad, weightsOffsets, dOffsets, hashSizeCumsum, indices, offsets);

    auto devWeightsCon = devWeights.contiguous();
    auto weightsGradCon = weightsGrad.contiguous();
    auto weightsOffsetsCon = weightsOffsets.contiguous();
    auto dOffsetsCon = dOffsets.contiguous().to(at::kLong);
    auto hashSizeCumsumCon = hashSizeCumsum.contiguous();
    auto indicesCon = indices.contiguous();
    auto offsetsCon = offsets.contiguous();
    auto indiceWeightsOptionalCon = indiceWeightsOptional.value_or(Tensor()).contiguous();
    auto bOffsetOptionalCon = bOffsetOptional.value_or(Tensor()).contiguous();
    auto vbeOutputOffsetsFeatureRankOptionalCon = vbeOutputOffsetsFeatureRankOptional.value_or(Tensor()).contiguous();
    auto vbeBOffsetsRankPerFeatureOptionalCon = vbeBOffsetsRankPerFeatureOptional.value_or(Tensor()).contiguous();
    at::Tensor out = at::zeros_like(devWeightsCon);
    const bool enableOptimization = true;
    EXEC_NPU_CMD(aclnnDenseEmbeddingCodegenLookupFunctionGrad,
                 devWeightsCon, weightsGradCon, weightsOffsetsCon,
                 dOffsetsCon, hashSizeCumsumCon, indicesCon, offsetsCon, indiceWeightsOptionalCon, bOffsetOptionalCon,
                 vbeOutputOffsetsFeatureRankOptionalCon, vbeBOffsetsRankPerFeatureOptionalCon,
                 totalD, maxD, totalHashSizeBits,
                 poolingMode, enableOptimization, outputDtypeOptional, maxB, maxBFeatureRank, vbeOutputSize, out);

    return {out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

// 通过继承torch::autograd::Function类实现前反向绑定

class DenseEmbeddingCodegenLookupFunction : public torch::autograd::Function<DenseEmbeddingCodegenLookupFunction> {
public:
    static at::Tensor forward(AutogradContext *ctx, const at::Tensor &devWeights, const at::Tensor weightsOffsets,
        const at::Tensor &dOffsets, const int64_t totalD, const int64_t maxD, const at::Tensor &hashSizeCumsum,
        const int64_t totalHashSizeBits, const at::Tensor &indices, const at::Tensor &offsets,
        const int64_t poolingMode, const std::optional<at::Tensor> &indiceWeightsOptional,
        const std::optional<at::Tensor> &featureRequiresGrad, const int64_t outputDtypeOptional,
        const std::optional<at::Tensor> &bOffsetOptional,
        const std::optional<at::Tensor> &vbeOutputOffsetsFeatureRankOptional,
        const std::optional<at::Tensor> &vbeBOffsetsRankPerFeatureOptional, const int64_t maxB,
        const int64_t maxBFeatureRank, const int64_t vbeOutputSize)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({ devWeights, weightsOffsets, dOffsets, hashSizeCumsum, indices, offsets,
            indiceWeightsOptional.value_or(Tensor()), bOffsetOptional.value_or(Tensor()),
            vbeOutputOffsetsFeatureRankOptional.value_or(Tensor()),
            vbeBOffsetsRankPerFeatureOptional.value_or(Tensor()), featureRequiresGrad.value_or(Tensor()) });
        ctx->saved_data["totalD"] = totalD;
        ctx->saved_data["maxD"] = maxD;
        ctx->saved_data["totalHashSizeBits"] = totalHashSizeBits;
        ctx->saved_data["poolingMode"] = poolingMode;
        ctx->saved_data["outputDtypeOptional"] = outputDtypeOptional;
        ctx->saved_data["maxB"] = maxB;
        ctx->saved_data["maxBFeatureRank"] = maxBFeatureRank;
        ctx->saved_data["vbeOutputSize"] = vbeOutputSize;
        return dense_embedding_codegen_lookup_function_impl_npu(devWeights, weightsOffsets, dOffsets, totalD, maxD,
            hashSizeCumsum, totalHashSizeBits, indices, offsets, poolingMode, indiceWeightsOptional,
            featureRequiresGrad, outputDtypeOptional, bOffsetOptional, vbeOutputOffsetsFeatureRankOptional,
            vbeBOffsetsRankPerFeatureOptional, maxB, maxBFeatureRank, vbeOutputSize);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto weightsGrad = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        auto devWeights = saved[0];
        auto weightsOffsets = saved[1];
        auto dOffsets = saved[2];
        auto hashSizeCumsum = saved[3];
        auto indices = saved[4];
        auto offsets = saved[5];
        auto indiceWeightsOptional = saved[6];
        auto bOffsetOptional = saved[7];
        auto vbeOutputOffsetsFeatureRankOptional = saved[8];
        auto vbeBOffsetsRankPerFeatureOptional = saved[9];
        auto featureRequiresGrad = saved[10];
        auto totalD = ctx->saved_data["totalD"].toInt();
        auto maxD = ctx->saved_data["maxD"].toInt();
        auto totalHashSizeBits = ctx->saved_data["totalHashSizeBits"].toInt();
        auto poolingMode = ctx->saved_data["poolingMode"].toInt();
        auto outputDtypeOptional = ctx->saved_data["outputDtypeOptional"].toInt();
        auto maxB = ctx->saved_data["maxB"].toInt();
        auto maxBFeatureRank = ctx->saved_data["maxBFeatureRank"].toInt();
        auto vbeOutputSize = ctx->saved_data["vbeOutputSize"].toInt();

        return dense_embedding_codegen_lookup_function_grad_impl_npu(devWeights, weightsGrad, weightsOffsets, dOffsets,
            totalD, maxD, hashSizeCumsum, totalHashSizeBits, indices, offsets, poolingMode, indiceWeightsOptional,
            featureRequiresGrad, outputDtypeOptional, bOffsetOptional, vbeOutputOffsetsFeatureRankOptional,
            vbeBOffsetsRankPerFeatureOptional, maxB, maxBFeatureRank, vbeOutputSize);
    }
};

// 使用的时候调用apply()方法
at::Tensor dense_embedding_codegen_lookup_function_impl_autograd(const at::Tensor &devWeights,
    const at::Tensor &weightsOffsets, const at::Tensor &dOffsets, const int64_t totalD, const int64_t maxD,
    const at::Tensor &hashSizeCumsum, const int64_t totalHashSizeBits, const at::Tensor &indices,
    const at::Tensor &offsets, const int64_t poolingMode, const std::optional<at::Tensor> &indiceWeightsOptional,
    const std::optional<at::Tensor> &featureRequiresGrad, const int64_t outputDtypeOptional,
    const std::optional<at::Tensor> &bOffsetOptional,
    const std::optional<at::Tensor> &vbeOutputOffsetsFeatureRankOptional,
    const std::optional<at::Tensor> &vbeBOffsetsRankPerFeatureOptional,
    const int64_t maxB,
    const int64_t maxBFeatureRank,
    const int64_t vbeOutputSize)
{
    return DenseEmbeddingCodegenLookupFunction::apply(devWeights, weightsOffsets, dOffsets, totalD, maxD,
        hashSizeCumsum, totalHashSizeBits, indices, offsets, poolingMode, indiceWeightsOptional, featureRequiresGrad,
        outputDtypeOptional, bOffsetOptional, vbeOutputOffsetsFeatureRankOptional, vbeBOffsetsRankPerFeatureOptional,
        maxB, maxBFeatureRank, vbeOutputSize);
}

// 在npu命名空间里注册dense_embedding_codegen_lookup_function_impl_npu
// 和dense_embedding_codegen_lookup_function_grad_impl_npu两个schema
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("dense_embedding_codegen_lookup_function("
        "Tensor devWeights, Tensor weightsOffsets, Tensor dOffsets, int totalD, int maxD, "
        "Tensor hashSizeCumsum, int totalHashSizeBits, Tensor indices,"
        "Tensor offsets, int poolingMode, Tensor? indiceWeightsOptional,"
        "Tensor? featureRequiresGrad, int outputDtypeOptional,"
        "Tensor? bOffsetOptional,"
        "Tensor? vbeOutputOffsetsFeatureRankOptional, "
        "Tensor? vbeBOffsetsRankPerFeatureOptional, "
        "int maxB, int maxBFeatureRank, "
        "int vbeOutputSize) -> Tensor");

    m.def("dense_embedding_codegen_lookup_function_grad("
        "Tensor devWeights, Tensor grad, Tensor weightsOffsets, "
        "Tensor dOffsets, int totalD, int maxD, Tensor hashSizeCumsum,int totalHashSizeBits,"
        "Tensor indices, Tensor offsets, int poolingMode, Tensor? indiceWeightsOptional, "
        "Tensor? featureRequiresGrad, int outputDtypeOptional, "
        "Tensor? bOffsetOptional,"
        "Tensor? vbeOutputOffsetsFeatureRankOptional, "
        "Tensor? vbeBOffsetsRankPerFeatureOptional, "
        "int maxB, int maxBFeatureRank, "
        "int vbeOutputSize) -> Tensor[]");
}

// 为NPU设备注册前反向实现，适用于require_grad = false的情况，前反向单独调用
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("dense_embedding_codegen_lookup_function", &dense_embedding_codegen_lookup_function_impl_npu);
    m.impl("dense_embedding_codegen_lookup_function_grad", &dense_embedding_codegen_lookup_function_grad_impl_npu);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("dense_embedding_codegen_lookup_function", &dense_embedding_codegen_lookup_function_impl_npu);
    m.impl("dense_embedding_codegen_lookup_function_grad", &dense_embedding_codegen_lookup_function_grad_impl_npu);
}
// 给op绑定NPU的自动求导实现，适用于require_grad = true的情况，自动调用反向
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("dense_embedding_codegen_lookup_function", &dense_embedding_codegen_lookup_function_impl_autograd);
}