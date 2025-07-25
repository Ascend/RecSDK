#ifndef IDS_MAPPER_H
#define IDS_MAPPER_H

#include <c10/util/flat_hash_map.h>
#include <omp.h>
#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>
#include <vector>


namespace hybrid {
constexpr int64_t MIN_IDS_LENGTH = 65536;
constexpr int64_t DOUBLE_INIT = 2;
constexpr int64_t PARTITION_LEN = 8192;
class IdsMapper : public torch::CustomClassHolder {
public:
    using Self = IdsMapper;
    explicit IdsMapper(int64_t initMaxIndex) : initMaxIndex(initMaxIndex){};
    IdsMapper(const IdsMapper& other) = delete;
    IdsMapper& operator=(const IdsMapper& other)
    {
        return *this;
    };
    std::tuple<at::Tensor, at::Tensor, at::Tensor> UniqueAndLookup(const torch::Tensor& globalIds);
    void UniqueAndLookupOut(const torch::Tensor& globalIds, const torch::Tensor& hashIndices,
                            const torch::Tensor& offset, const torch::Tensor& unique,
                            const torch::Tensor& uniqueInverse, const torch::Tensor& uniqueOffset, int64_t tableId);

    std::unique_ptr<std::vector<int64_t>> AllocFullHashMap()
    {
        std::lock_guard<std::mutex> lock(allocMute);
        std::unique_ptr<std::vector<int64_t>> oneMap;
        if (fullHashMapQue.size() == 0) {
            oneMap = std::make_unique<std::vector<int64_t>>(maxIndex * DOUBLE_INIT, -1);
        } else {
            oneMap = std::move(fullHashMapQue.front());
            fullHashMapQue.pop();
        }
        if (oneMap->capacity() <= maxIndex) {
            oneMap->resize(maxIndex * DOUBLE_INIT, -1);
        }
        return oneMap;
    }

    void DeallocFullHashMap(std::unique_ptr<std::vector<int64_t>> oneMap)
    {
        std::lock_guard<std::mutex> lock(allocMute);
        fullHashMapQue.push(std::move(oneMap));
    }

private:
    
    void UniqueProcessing(const torch::Tensor& hashIndices, const torch::Tensor& offset,
        const torch::Tensor& unique, const torch::Tensor& uniqueInverse,
        const torch::Tensor& uniqueOffset, int64_t tableId);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> FindOrInsertHighPrecison(const            torch::Tensor& global_ids);
    ska::flat_hash_map<int64_t, int64_t> ids2indicesMap;

    int numThread;
    std::queue<std::unique_ptr<std::vector<int64_t>>> fullHashMapQue;

    int64_t maxIndex = 0;
    int64_t initMaxIndex;

    std::mutex insertMute;
    std::mutex allocMute;
    // Profiling Record
    at::ThreadLocalState state;
};
}  // namespace hybrid
#endif