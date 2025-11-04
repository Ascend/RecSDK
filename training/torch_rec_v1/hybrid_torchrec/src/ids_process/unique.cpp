/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "unique.h"

#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/zeros.h>
#include <ATen/record_function.h>
#include <c10/util/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <tuple>
#include <vector>

using at::indexing::Slice;
namespace hybrid {
template <typename scalar_t, bool equal_nan>
struct IsUnique {};

template <typename scalar_t>
struct IsUnique<scalar_t, false> {
    inline bool operator()(scalar_t* dataPtr, int64_t i)
    {
        if (i == 0) {
            return true;
        }
        return c10::load(&dataPtr[i]) != c10::load(&dataPtr[i - 1]);
    }
};

template <typename scalar_t>
struct IsUnique<scalar_t, true> {
    inline bool operator()(scalar_t* dataPtr, int64_t i)
    {
        if (i == 0) {
            return true;
        }
        return (c10::load(&dataPtr[i]) != c10::load(&dataPtr[i - 1])) &&
               !(_isnan(dataPtr[i]) && _isnan(dataPtr[i - 1]));
    }
};

template <typename scalar_t, typename CompareOp>
std::tuple<at::Tensor, at::Tensor> UniqueCpuSortedTemplate(const at::Tensor& self, CompareOp isUnique)
{
    const at::Tensor& input = self.contiguous();

    int64_t numel = input.numel();
    at::Tensor inverseIndices = at::empty({numel}, self.options().dtype(c10::kLong));
    if (numel == 0) {
        return std::make_tuple(at::empty({0}, self.options().dtype(c10::kLong)),
                               at::empty({0}, self.options().dtype(c10::kLong)));
    }

    auto inputFlattened = input.flatten();

    auto [inputSorted, indices] = inputFlattened.sort();

    scalar_t* inputSortedData = inputSorted.data_ptr<scalar_t>();
    int64_t* indicesData = indices.data_ptr<int64_t>();

    int numThreads = at::get_num_threads();

    std::vector<int64_t> uniqueCountThread(numThreads, 0);
    std::vector<int64_t> offsetThread(numThreads, 0);

    const int64_t grainSize = at::internal::GRAIN_SIZE;

    // calculate unique count from each thread
    at::parallel_for(0, numel, grainSize, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        for (const auto i : c10::irange(begin, end)) {
            if (isUnique(inputSortedData, i)) {
                uniqueCountThread[tid]++;
            }
        }
    });

    // calculate thread offset in output and
    // `uniqueCount` records total count of uniques at last
    int64_t uniqueCount = 0;
    for (const auto t : c10::irange(numThreads)) {
        offsetThread[t] = uniqueCount;
        uniqueCount += uniqueCountThread[t];
    }

    at::Tensor output = at::empty({uniqueCount}, self.options());
    scalar_t* outputData = output.data_ptr<scalar_t>();

    int64_t* inverseIndicesData = inverseIndices.data_ptr<int64_t>();

    at::parallel_for(0, numel, grainSize, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        int64_t offset = offsetThread[tid];
        for (const auto i : c10::irange(begin, end)) {
            if (isUnique(inputSortedData, i)) {
                outputData[offset] = c10::load(&inputSortedData[i]);
                offset++;
            }

            int64_t inverseIndex = offset - 1;
            int64_t perm = indicesData[i];
            inverseIndicesData[perm] = inverseIndex;
        }
    });
    return std::make_tuple(output, inverseIndices);
}

std::tuple<at::Tensor, at::Tensor> UniqueParallel(const at::Tensor& ids)
{
    RECORD_FUNCTION(c10::str("hybrid::UniqueParallel"), c10::ArrayRef<const c10::IValue>({ids.numel()}));
    at::Tensor output;
    at::Tensor inverseIndices;
    std::tie(output, inverseIndices) = UniqueCpuSortedTemplate<int64_t>(ids, IsUnique<int64_t, false>());
    return {output, inverseIndices};
}

}  // namespace hybrid
