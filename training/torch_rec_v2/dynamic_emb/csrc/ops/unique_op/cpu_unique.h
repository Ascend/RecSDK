#ifndef CPU_UNIQUE_H
#define CPU_UNIQUE_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <torch/extension.h>

__inline__ uint64_t MurmurHash3ForUnique(uint64_t const& key)
{
    uint64_t k = key;
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

inline int64_t RoundUpToPowerOfTwo(int64_t num)
{
    if ((num & (num - 1)) == 0) {
        return num;
    }

    int64_t rounded = 1;
    while (rounded < num) {
        rounded <<= 1;
    }

    return rounded;
}


template <typename KeyType>
class CPUUniqueHash {
public:
    void ComputeUnique(const KeyType* keys, int64_t num)
    {
        if (num == 0) {
            uniqueNum = 0;
            numKeys = 0;
            return;
        }

        numKeys = num;
        int64_t hashTableSize = RoundUpToPowerOfTwo(num);
        const uint64_t mask = static_cast<uint64_t>(hashTableSize - 1);
        std::vector<Entry> table(hashTableSize);
        counts.resize(num, 0);
        inverse.resize(num);
        cpuKeys.resize(num);
        int64_t counter = 0;

        for (int64_t i = 0; i < num; ++i) {
            KeyType key = keys[i];
            uint64_t hashValue = MurmurHash3ForUnique(static_cast<uint64_t>(key));
            uint64_t index = hashValue & mask;
            int64_t probeCount = 0;

            while (true) {
                auto& entry = table[index];
                if (!entry.used) {
                    entry.used = true;
                    entry.key = key;
                    entry.index = counter;
                    cpuKeys[counter] = key;
                    inverse[i] = counter;
                    counts[counter] = 1;
                    counter++;
                    break;
                }
                if (entry.key == key) {
                    inverse[i] = entry.index;
                    counts[entry.index]++;
                    break;
                }
                index = (index + 1) & mask;
                probeCount++;
                if (probeCount >= num) {
                    throw std::runtime_error("Probe failed, hash table overflow!");
                }
            }
        }

        uniqueNum = counter;
        counts.resize(counter);
        cpuKeys.resize(counter);
    }

    const std::vector<KeyType>& GetUniqueKeys() const
    {
        return cpuKeys;
    }

    const std::vector<int64_t>& GetKeyCounts() const
    {
        return counts;
    }

    const std::vector<int64_t>& GetInverseIndices() const
    {
        return inverse;
    }

    int64_t GetUniqueCount() const
    {
        return uniqueNum;
    }

    int64_t GetTotalCount() const
    {
        return numKeys;
    }

private:
    struct Entry {
        KeyType key{0};
        int64_t index{0};
        bool used{false};
    };

    std::vector<KeyType> cpuKeys;
    std::vector<int64_t> inverse;
    std::vector<int64_t> counts;
    int64_t uniqueNum{0};
    int64_t numKeys{0};
};

struct DedupResult {
    at::Tensor uniqueElements;
    at::Tensor counts;
    at::Tensor reverseIndices;
    int64_t uniqueCount;
};

template <typename KeyType>
class HashDeduplicator {
public:
    HashDeduplicator() = default;

    DedupResult deduplicate(const at::Tensor& data)
    {
        const int64_t n = data.numel();
        if (n == 0) {
            DedupResult emptyRes;
            emptyRes.uniqueCount = 0;
            return emptyRes;
        }

        const KeyType* inputPtr = data.data_ptr<KeyType>();
        hashObj.ComputeUnique(inputPtr, n);

        const auto& uniqueKeys = hashObj.GetUniqueKeys();
        const auto& keyCounts = hashObj.GetKeyCounts();
        const auto& inverseindices = hashObj.GetInverseIndices();
        const int64_t uniqueNum = hashObj.GetUniqueCount();
        auto optionsInt64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        DedupResult res;

        auto uniqueTensorOptions = torch::TensorOptions().dtype(data.dtype()).device(torch::kCPU);
        res.uniqueElements = at::empty({uniqueNum}, uniqueTensorOptions);
        if (uniqueNum > 0) {
            size_t dataSize = uniqueNum * sizeof(KeyType);
            errno_t ret = memcpy_s(res.uniqueElements.data_ptr<KeyType>(), dataSize, uniqueKeys.data(), dataSize);
            TORCH_CHECK(ret == EOK, "UniqueIndicesRange memcpy_s failed, ret = ", ret);
        }
        res.counts = at::tensor(keyCounts, optionsInt64);
        res.reverseIndices = at::tensor(inverseindices, optionsInt64);
        res.uniqueCount = static_cast<int64_t>(uniqueNum);

        return res;
    }

private:
    CPUUniqueHash<KeyType> hashObj;
};

#endif  // CPU_UNIQUE_H
