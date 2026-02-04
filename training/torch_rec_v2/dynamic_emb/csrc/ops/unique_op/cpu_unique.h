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

class CPUUniqueHash {
public:
    void ComputeUnique(const int64_t* keys, int64_t num)
    {
        if (num == 0) {
            uniqueNum = 0;
            numKeys = 0;
            return;
        }

        numKeys = num;
        std::vector<Entry> table(num);
        counts.resize(num, 0);
        inverse.resize(num);
        cpuKeys.resize(num);

        int64_t counter = 0;
        std::hash<int64_t> hashFunc;

        for (int64_t i = 0; i < num; ++i) {
            int64_t key = keys[i];
            size_t hashValue = hashFunc(key);
            int64_t index = static_cast<int64_t>(hashValue) % num;
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
                index = (index + 1) % num;
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

    const std::vector<int64_t>& GetUniqueKeys() const
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
        int64_t key{0};
        int64_t index{0};
        bool used{false};
    };

    std::vector<int64_t> cpuKeys;
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

        const int64_t* inputPtr = data.data_ptr<int64_t>();
        hashObj.ComputeUnique(inputPtr, n);

        const auto& uniqueKeys = hashObj.GetUniqueKeys();
        const auto& keyCounts = hashObj.GetKeyCounts();
        const auto& inverseindices = hashObj.GetInverseIndices();
        const int64_t uniqueNum = hashObj.GetUniqueCount();

        auto optionsInt64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        DedupResult res;

        res.uniqueElements = at::tensor(uniqueKeys, optionsInt64);
        res.counts = at::tensor(keyCounts, optionsInt64);
        res.reverseIndices = at::tensor(inverseindices, optionsInt64);
        res.uniqueCount = static_cast<int64_t>(uniqueNum);

        return res;
    }

private:
    CPUUniqueHash hashObj;
};

#endif  // CPU_UNIQUE_H
