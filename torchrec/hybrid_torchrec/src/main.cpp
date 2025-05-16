#include <ATen/core/TensorBody.h>
#include <ATen/ops/concat.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/zeros.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "ids_mapper.h"
#include "unique.h"

constexpr int TEST_NUM = 100;
void TestMapper()
{
    const auto testId = at::randint(0, 40000000, {1700000});
    auto mapper = hybrid::IdsMapper(40000000);
    std::cout << " insert start " << std::endl;
    mapper.UniqueAndLookup(testId, false);
    std::cout << " insert end " << std::endl;
    for (int i = 0; i < TEST_NUM; i++) {
        mapper.UniqueAndLookup(testId, false);
    }
    auto beforeTime = std::chrono::steady_clock::now();
    for (int i = 0; i < TEST_NUM; i++) {
        mapper.UniqueAndLookup(testId, false);
    }
    auto afterTime = std::chrono::steady_clock::now();
    // 秒
    double durationSecond = std::chrono::duration<double, std::micro>(afterTime - beforeTime).count();
    std::cout << "UniqueAndLookup总耗时: " << durationSecond / TEST_NUM << "us" << std::endl;
}

void TestMapperOut()
{
    const auto testId = at::randint(0, 40000000, {1700000});
    const auto testId1 = at::randint(0, 40000000, {1700000});
    const auto testId2 = at::randint(0, 40000000, {1700000});
    const auto testIds = at::concat({testId, testId1, testId2});
    const auto hashIndices = at::empty_like(testIds);
    std::vector<int64_t> v = {0, 1700000, 1700000 * 2, 1700000 * 3};
    auto offsets = at::from_blob(v.data(), v.size(), testIds.options());
    const auto unique = at::empty_like(testIds);
    const auto uniqueInverse = at::empty_like(testIds);
    const auto uniqueOffset = at::zeros({4}, offsets.options());

    auto mapper1 = hybrid::IdsMapper(40000000);
    auto mapper2 = hybrid::IdsMapper(40000000);
    auto mapper3 = hybrid::IdsMapper(40000000);
    int firstTable = 0;
    int secondTable = 1;
    int thirdTable = 2;
    int tableNum = 3;
    std::cout << " insert start " << std::endl;
    mapper1.UniqueAndLookupOut(testIds, hashIndices, offsets, unique, uniqueInverse, uniqueOffset, firstTable);
    mapper2.UniqueAndLookupOut(testIds, hashIndices, offsets, unique, uniqueInverse, uniqueOffset, secondTable);
    mapper3.UniqueAndLookupOut(testIds, hashIndices, offsets, unique, uniqueInverse, uniqueOffset, thirdTable);
    std::cout << " insert end " << std::endl;
    auto beforeTime = std::chrono::steady_clock::now();
    for (int i = 0; i < TEST_NUM; i++) {
        mapper1.UniqueAndLookupOut(testIds, hashIndices, offsets, unique, uniqueInverse, uniqueOffset, firstTable);
        mapper2.UniqueAndLookupOut(testIds, hashIndices, offsets, unique, uniqueInverse, uniqueOffset, secondTable);
        mapper3.UniqueAndLookupOut(testIds, hashIndices, offsets, unique, uniqueInverse, uniqueOffset, thirdTable);
    }
    auto afterTime = std::chrono::steady_clock::now();
    // 秒
    double durationSecond = std::chrono::duration<double, std::micro>(afterTime - beforeTime).count();
    std::cout << "UniqueAndLookup总耗时: " << (durationSecond / TEST_NUM / tableNum) << "us" << std::endl;
}

void TestUnique()
{
    const auto testId = at::randint(0, 40000000, {1700000});
    for (int i = 0; i < TEST_NUM; i++) {
        hybrid::UniqueParallel(testId);
    }

    auto beforeTime = std::chrono::steady_clock::now();
    for (int i = 0; i < TEST_NUM; i++) {
        hybrid::UniqueParallel(testId);
    }
    auto afterTime = std::chrono::steady_clock::now();
    // 秒
    double durationSecond = std::chrono::duration<double, std::micro>(afterTime - beforeTime).count();
    std::cout << "unique总耗时: " << durationSecond / TEST_NUM << "us" << std::endl;
}

void TestSort()
{
    const auto testId = at::randint(0, 40000000, {1700000});
    for (int i = 0; i < TEST_NUM; i++) {
        at::sort(testId);
    }
    std::vector<std::thread> threads;

    auto beforeTime = std::chrono::steady_clock::now();
    for (int i = 0; i < TEST_NUM; i++) {
        at::sort(testId);
    }

    auto afterTime = std::chrono::steady_clock::now();
    // 秒
    double durationSecond = std::chrono::duration<double, std::micro>(afterTime - beforeTime).count();
    std::cout << "sort总耗时: " << durationSecond / TEST_NUM << "us" << std::endl;
}

int main()
{
    TestMapperOut();
    TestMapper();
    TestUnique();
    TestSort();
    return 0;
}