#ifndef HYBRID_UNIQUE_H
#define HYBRID_UNIQUE_H
#include <omp.h>
#include <torch/torch.h>


namespace hybrid {
std::tuple<at::Tensor, at::Tensor> UniqueParallel(const at::Tensor& ids);
void UniqueParallelMutilTensoOut(const torch::Tensor& hashIndices, const torch::Tensor& offset,
                                 const torch::Tensor& unique, const torch::Tensor& uniqueInverse,
                                 const torch::Tensor& uniqueOffset, int64_t tensorI);
}  // namespace hybrid
#endif