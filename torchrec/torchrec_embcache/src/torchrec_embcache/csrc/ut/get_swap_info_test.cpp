/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include "embedding_cache/swap_manager.h"

using namespace Embcache;

constexpr int BATCH_SIZE = 100000;
constexpr int INT10 = 10;
constexpr int NUM_SIZE = 100;

std::vector<int64_t> GenRandKeys(int numKeys, int64_t limit = std::numeric_limits<int64_t>::max())
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> distrib(0, limit);

    std::vector<int64_t> keys(numKeys);
    for (int i = 0; i < numKeys; ++i) {
        keys[i] = distrib(gen);
    }

    return keys;
}

class Gen {
public:
    Gen(int keyNum) : keyNum(keyNum)
    {
        allKeys = GenRandKeys(keyNum);
    }

    std::vector<int64_t> GenKeys(int num)
    {
        std::vector<int64_t> idx = GenRandKeys(num, keyNum - 1);
        std::vector<int64_t> ret;
        for (int64_t i : idx) {
            ret.push_back(allKeys[i]);
        }
        return ret;
    }

private:
    std::vector<int64_t> allKeys;
    int keyNum;
};

#ifdef DEBUG_PRINT

// 辅助函数：打印 vector
template <typename T>
void PrintVector(const std::vector<T>& vec, const std::string& name)
{
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// 辅助函数：打印 unordered_map
template <typename K, typename V>
void PrintUnorderedMap(const std::unordered_map<K, V>& map, const std::string& name)
{
    std::cout << name << ": {";
    int i = 0;
    for (const auto& pair : map) {
        std::cout << pair.first << ": " << pair.second;
        if (i < map.size() - 1) {
            std::cout << ", ";
        }
        i++;
    }
    std::cout << "}" << std::endl;
}

// 打印 SwapManager 状态
void PrintSwapManagerState(const std::string& status, const std::vector<int64_t>& cache,
                           const std::unordered_map<int64_t, int64_t>& key2off)
{
    std::cout << status << std::endl;
    PrintVector(cache, "Cache");
    PrintUnorderedMap(key2off, "Key2Off");
    std::cout << std::endl;
}

// 打印 Swap In/Out 向量
void PrintSwapIOVectors(const std::vector<int64_t>& swapoutKeys, const std::vector<int64_t>& swapoutOffs,
                        const std::vector<int64_t>& swapinKeys, const std::vector<int64_t>& swapinOffs)
{
    PrintVector(swapoutKeys, "Swapout Keys");
    PrintVector(swapoutOffs, "Swapout Offs");
    PrintVector(swapinKeys, "Swapin Keys");
    PrintVector(swapinOffs, "Swapin Offs");
    std::cout << std::endl;
}

#endif

void SwapManagerTest(int64_t batchSize, int64_t cacheSize, int testNum)
{
    Gen gen(cacheSize * INT10);
    SwapManager swapManager(cacheSize);

    std::vector<int64_t> cache(cacheSize);
    std::unordered_map<int64_t, int64_t> key2off;

    for (int t = 0; t < testNum; t++) {
        std::cout << "\n===================== Testing batch " << t << " =====================" << std::endl;
        auto keys = gen.GenKeys(batchSize);
        std::unordered_set<int64_t> keySet(keys.begin(), keys.end());

#ifdef DEBUG_PRINT
        PrintVector(keys, "Keys");
        std::cout << std::endl;
        PrintSwapManagerState("Before GetSwapVecsAndKey2Offset:", cache, key2off);
#endif

        auto tp = swapManager.ComputeSwapInfo(keys);
        auto swapoutKeys = std::get<0>(tp);
        auto swapoutOffs = std::get<1>(tp);
        auto swapinKeys = std::get<2>(tp);
        auto swapinOffs = std::get<3>(tp);
        auto batchOffs = std::get<4>(tp);

#ifdef DEBUG_PRINT
        PrintSwapIOVectors(swapoutKeys, swapoutOffs, swapinKeys, swapinOffs);
#endif

        // 1. key 和 off size 相同
        assert(swapoutKeys.size() == swapoutOffs.size());
        assert(swapinKeys.size() == swapinOffs.size());

        std::cout << "swapout size: " << swapoutKeys.size() << std::endl;
        std::cout << "swapin size: " << swapinKeys.size() << std::endl;
        std::cout << std::endl;

        // 2. 执行 swapout，验证 swapoutKeys 和 swapoutOffs 都在 cache 中，
        // 且 swapoutKeys 不在 keys 中，且 off 都在范围内
        for (int64_t i = 0; i < swapoutKeys.size(); i++) {
            auto key = swapoutKeys[i];
            auto off = swapoutOffs[i];

            assert(off < cacheSize);
            assert(cache[off] == key);
            assert(key2off.find(key) != key2off.end());
            assert(keySet.find(key) == keySet.end());

            cache[off] = 0;
            key2off.erase(key);
        }

#ifdef DEBUG_PRINT
        PrintSwapManagerState("After Swapout:", cache, key2off);
#endif

        // 3. 执行 swapin，验证 swapinKeys 都在 keys 中，
        // 且 swapinKeys 和 swapoutOffs 都不在 cache 中，且 off 都在范围内
        for (int64_t i = 0; i < swapinKeys.size(); i++) {
            auto key = swapinKeys[i];
            auto off = swapinOffs[i];

            assert(off < cacheSize);
            assert(keySet.find(key) != keySet.end());
            if (key2off.find(key) != key2off.end()) {
                std::cout << i << ' ' << key2off[key] << std::endl;
            }
            assert(key2off.find(key) == key2off.end());

            cache[off] = key;
            key2off[key] = off;
        }

#ifdef DEBUG_PRINT
        PrintSwapManagerState("After Swapin:", cache, key2off);
#endif

        // 4. 验证 keys 都在 cache 中，且 key 都转化成 off
        for (int64_t i = 0; i < batchOffs.size(); i++) {
            auto key = keys[i];
            auto off = batchOffs[i];
            assert(key2off.find(key) != key2off.end());
            assert(cache[off] == key);
            assert(key2off[key] == off);
        }

        std::cout << "Test " << t << " passed!" << std::endl;
    }
}

int main()
{
    int64_t batchSize;
    int64_t cacheSize;
    int testNum;

    batchSize = 0;
    cacheSize = 0;
    testNum = NUM_SIZE;
    SwapManagerTest(batchSize, cacheSize, testNum);

    batchSize = 1;
    cacheSize = 1;
    testNum = NUM_SIZE;
    SwapManagerTest(batchSize, cacheSize, testNum);

    batchSize = BATCH_SIZE;
    cacheSize = BATCH_SIZE;
    testNum = NUM_SIZE;
    SwapManagerTest(batchSize, cacheSize, testNum);
    return 0;
}