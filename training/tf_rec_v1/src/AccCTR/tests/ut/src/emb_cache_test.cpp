/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 ==============================================================================*/

#include "emb_cache_test.h"

#include <cmath>
#include <sstream>

#include "common.h"
#include "common/util/error_code.h"

using namespace std;
using namespace ock::ctr;

FactoryPtr factory;
EmbCacheManagerPtr embCache = nullptr;

std::vector<uint64_t> GenKeys(uint64_t n, uint32_t seed = 0, uint64_t min = 0, uint64_t max = UINT64_MAX)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<uint64_t> distribution(min, max);
    std::vector<uint64_t> data(n);
    for (uint64_t& x : data) {
        x = distribution(generator);
    }
    sort(data.begin(), data.end());
    data.erase(unique(data.begin(), data.end()), data.end());
    return data;
}

std::vector<uint64_t> GenUniqueKeys(uint64_t n)
{
    std::vector<uint64_t> data(n);
    for (uint64_t i = 0; i < n; i++) {
        data[i] = i;
    }
    return data;
}

EmbCacheManagerPtr EmbCacheTest::SimpleCreateTable(std::string tableName, uint32_t hostVocabSize,
                                                   uint32_t embeddingSize, uint32_t extEmbeddingSize,
                                                   uint32_t devVocabSize, pair<float, float> normalPara,
                                                   float constPara)
{
    factory->CreateEmbCacheManager(embCache);
    EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    EmbCache::NormalInitializerInfo normalInitializerInfo(normalPara.first, normalPara.second, 0, 1.0);
    std::string normalInitializeName = "random_normal_initializer";
    EmbCache::InitializerInfo normalInitializeInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo);

    EmbCache::ConstantInitializerInfo constantInitializerInfo(constPara, 1.0);
    std::string constantInitializeName = "constant_initializer";

    std::vector<EmbCache::InitializerInfo> initializeInfos(extEmbeddingSize / embeddingSize);
    initializeInfos[0] = normalInitializeInfo;
    for (uint64_t i = 1; i < initializeInfos.size(); i++) {
        initializeInfos[i] = EmbCache::InitializerInfo(constantInitializeName, embeddingSize * i, embeddingSize,
                                                       constantInitializerInfo);
    }
    int ret = embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize, 1);
    if (ret != H_OK) {
        string msg = "CreateCacheForTable Failed. ret: " + std::to_string(ret);
        CTRLog(CTRLogLevel::ERROR, msg.c_str());
        return nullptr;
    }
    return embCache;
}

EmbCacheManagerPtr EmbCacheTest::ConstZeroCreateTable(std::string tableName, uint32_t hostVocabSize,
                                                      uint32_t embeddingSize, uint32_t extEmbeddingSize,
                                                      uint32_t devVocabSize, uint64_t prefillBufferSize,
                                                      uint8_t prefillThreadNum)
{
    factory->CreateEmbCacheManager(embCache);
    EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    EmbCache::ConstantInitializerInfo constantInitializerInfo(0.0, 1.0);
    std::string constantInitializeName = "constant_initializer";

    std::vector<EmbCache::InitializerInfo> initializeInfos = {
        EmbCache::InitializerInfo(constantInitializeName, 0, extEmbeddingSize, constantInitializerInfo)};
    int ret = embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, prefillBufferSize, prefillThreadNum);
    if (ret != H_OK) {
        string msg = "CreateCacheForTable Failed. ret: " + std::to_string(ret);
        CTRLog(CTRLogLevel::ERROR, msg.c_str());
        return nullptr;
    }
    return embCache;
}

void EmbCacheTest::SetUpTestCase()
{
    Factory::Create(factory);
    factory->SetExternalLogFuncInner(CTRLog);
}

void EmbCacheTest::TearDownTestCase() {}

void EmbCacheTest::SetUp() {}

void EmbCacheTest::TearDown()
{
    if (embCache != nullptr) {
        embCache->Destroy();
        embCache = nullptr;
    }
}

TEST_F(EmbCacheTest, ConstantInitializerInfo)
{
    CTRLog(CTRLogLevel::INFO, "===========ConstantInitializerInfo start=============");

    // 正确初始化ConstantInitializerInfo结构体，无日志信息反馈
    EmbCache::ConstantInitializerInfo constantInitializerInfo(0.233, 1.0);
    CTRLog(CTRLogLevel::INFO, "===========ConstantInitializerInfo end=============");
}

TEST_F(EmbCacheTest, NormalInitializerInfo)
{
    CTRLog(CTRLogLevel::INFO, "===========NormalInitializerInfo start=============");
    // 正确初始化NormalInitializerInfo结构体，无日志信息反馈
    EmbCache::NormalInitializerInfo normalInitializerInfo(0, 0.05, 0, 1.0);
    // 标准差负值数学意义不明，传入负值问题用户自己承担
    EmbCache::NormalInitializerInfo normalInitializerInfo_ne_dev(0, -0.05, 0, 1.0);
    CTRLog(CTRLogLevel::INFO, "===========NormalInitializerInfo end=============");
}

TEST_F(EmbCacheTest, InitializerInfo)
{
    CTRLog(CTRLogLevel::INFO, "===========InitializerInfo start=============");
    uint32_t embeddingSize = 13;

    EmbCache::NormalInitializerInfo normalInitializerInfo(0, 0.05, 0, 1.0);
    EmbCache::ConstantInitializerInfo constantInitializerInfo(0.233, 1.0);

    // 传入的std::string不为"constant_initializer" 日志打印"Invalid Initializer Type."
    std::string not_a_initializer_name = "not_a_initializer_name";
    EmbCache::InitializerInfo constantInitializeInfo =
        EmbCache::InitializerInfo(not_a_initializer_name, embeddingSize, embeddingSize, constantInitializerInfo);

    // 传入的std::string不为"constant_initializer" 日志打印"Invalid Initializer Type."
    not_a_initializer_name = "";
    constantInitializeInfo =
        EmbCache::InitializerInfo(not_a_initializer_name, embeddingSize, embeddingSize, constantInitializerInfo);

    // 正确初始化InitializeInfo结构体，无日志信息反馈
    std::string constantInitializeName = "constant_initializer";
    constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize + 1, constantInitializerInfo);

    // 传入的std::string不为"random_normal_initializer"或truncated_normal_initializer 日志打印"Invalid Initializer
    // Type."
    not_a_initializer_name = "not_a_initializer_name";
    EmbCache::InitializerInfo normalInitializeInfo =
        EmbCache::InitializerInfo(not_a_initializer_name, embeddingSize, embeddingSize, normalInitializerInfo);

    // 传入的std::string不为"random_normal_initializer"或truncated_normal_initializer 日志打印"Invalid Initializer
    // Type."
    not_a_initializer_name = "";
    normalInitializeInfo =
        EmbCache::InitializerInfo(not_a_initializer_name, embeddingSize, embeddingSize, normalInitializerInfo);

    // 正确初始化InitializeInfo结构体，无日志信息反馈
    std::string normalInitializeName = "random_normal_initializer";
    normalInitializeInfo = EmbCache::InitializerInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo);

    // 正确初始化InitializeInfo结构体，无日志信息反馈
    std::string truncatedNormalInitializeName = "truncated_normal_initializer";
    EmbCache::InitializerInfo truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, 0, embeddingSize, normalInitializerInfo);

    CTRLog(CTRLogLevel::INFO, "===========InitializerInfo end=============");
}

TEST_F(EmbCacheTest, EmbCacheInfo)
{
    CTRLog(CTRLogLevel::INFO, "===========EmbCacheInfo start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    // 正确初始化EmbCacheInfo结构体，无日志信息反馈
    EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    CTRLog(CTRLogLevel::INFO, "===========EmbCacheInfo end=============");
}

TEST_F(EmbCacheTest, CreateCacheForTable)
{
    factory->CreateEmbCacheManager(embCache);
    CTRLog(CTRLogLevel::INFO, "===========CreateCacheForTable start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, {}, -1, hostVocabSize), H_INITIALIZER_INVALID);

    EmbCache::NormalInitializerInfo normalInitializerInfo(0, 0.05, 0, 1.0);
    std::string normalInitializeName = "random_normal_initializer";
    EmbCache::InitializerInfo normalInitializeInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo);

    // 空initializer 日志打印出"Initializer is nullptr"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, {{}, {}}, -1, hostVocabSize), H_INITIALIZER_INVALID);

    normalInitializeInfo.initializer = nullptr;
    // 空initializer 日志打印出"Initializer is nullptr"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, {normalInitializeInfo}, -1, hostVocabSize),
              H_INITIALIZER_INVALID);

    normalInitializeInfo = EmbCache::InitializerInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo);
    EmbCache::ConstantInitializerInfo constantInitializerInfo(0.233, 1.0);
    std::string constantInitializeName = "constant_initializer";
    EmbCache::InitializerInfo constantInitializeInfo(constantInitializeName, embeddingSize, embeddingSize + 1,
                                                     constantInitializerInfo);
    std::vector<EmbCache::InitializerInfo> initializeInfos = {normalInitializeInfo, constantInitializeInfo};

    // initializerInfos的区间之间有重叠或者遗漏 日志打印出"Initializers got coverage problems"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_INITIALIZER_INVALID);

    constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize + 1, embeddingSize, constantInitializerInfo);
    initializeInfos = {normalInitializeInfo, constantInitializeInfo};
    // initializerInfos的区间之间有重叠或者遗漏 日志打印出"Initializers got coverage problems"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_INITIALIZER_INVALID);

    embCacheInfo.extEmbeddingSize = extEmbeddingSize;
    std::string not_a_initializer_name = "not_a_initializer_name";
    constantInitializeInfo =
        EmbCache::InitializerInfo(not_a_initializer_name, embeddingSize, embeddingSize, constantInitializerInfo);
    initializeInfos = {normalInitializeInfo, constantInitializeInfo};

    // 传入的Initializer的name不符要求 日志打印出"Invalid Initializer Type.\nInitializer is nullptr"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_INITIALIZER_INVALID);

    constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize, constantInitializerInfo);
    initializeInfos = {normalInitializeInfo, constantInitializeInfo};

    embCacheInfo.extEmbeddingSize++;

    // 传入的embInfo中的传入的extEmbeddingSize并非embeddingSize的整数倍 日志打印出"extEmbeddingSize = embeddingSize +
    // optimizerSize, which is divisible by embeddingSize"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize),
              H_EXT_EMBEDDING_SIZE_INVALID);

    embCacheInfo.maxCacheSize = 100;
    // maxCacheSize>vocabSize 日志打印出"vocabSize must be greater than or equal to maxCacheSize"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize),
              H_HOST_VOCAB_SIZE_TOO_SMALL);
    embCacheInfo.maxCacheSize = devVocabSize;

    embCacheInfo.extEmbeddingSize = 0;
    // extEmbeddingSize为0 日志打印出"size must be positive"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_SIZE_ZERO);
    embCacheInfo.extEmbeddingSize = extEmbeddingSize;

    embCacheInfo.embeddingSize = 0;
    // embeddingSize为0 日志打印出"size must be positive"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_SIZE_ZERO);
    embCacheInfo.embeddingSize = embeddingSize;

    embCacheInfo.vocabSize = 0;
    // vocabSize为0 日志打印出"size must be positive"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_SIZE_ZERO);
    embCacheInfo.vocabSize = hostVocabSize;

    embCacheInfo.maxCacheSize = 0;
    // maxCacheSize为0 日志打印出"size must be positive"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_SIZE_ZERO);
    embCacheInfo.maxCacheSize = devVocabSize;

    embCacheInfo.tableName = "";
    // 传入的tableName空 日志打印出"tableName can not be empty"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_TABLE_NAME_EMPTY);

    embCacheInfo.tableName =
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "0000000001000000000100000000010001";
    // 传入的tableName长度正好为长度上限1024
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_OK);

    embCacheInfo.tableName =
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100000000010000000001000000000100000000010000000001000000000100000000010000000001"
        "00000000010000000001000000000100012";
    // 传入的tableName长度为1025超过了长度上限
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_TABLE_NAME_TOO_LONG);
    embCacheInfo.tableName = tableName;

    // 正常创建 日志中不会打印异常信息
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_OK);

    // 重复创建同名Table 日志打印出"This table has already been created"
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize),
              H_TABLE_CREATE_DUPLICATE);
    embCache->Destroy();

    // Destroy后仍能正常创建 日志中不会打印异常信息
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize), H_OK);
    embCache->Destroy();

    // prefill单线程
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 3, 1), H_OK);
    embCache->Destroy();

    // prefill多线程
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 3, 3), H_OK);
    embCache->Destroy();

    // prefill多线程
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 3, 0), H_THREAD_NUM_ERROR);
    embCache->Destroy();

    // prefill过多线程
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 3, 10000), H_THREAD_NUM_ERROR);
    embCache->Destroy();

    // prefill 正常buffersize
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 3, 1), H_OK);
    embCache->Destroy();

    // prefill 超大buffersize
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 10, 1), H_PREFILL_BUFFER_SIZE_INVALID);
    embCache->Destroy();

    // prefill 0buffersize
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, 0, 1), H_PREFILL_BUFFER_SIZE_INVALID);
    CTRLog(CTRLogLevel::INFO, "===========CreateCacheForTable end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP_ADDRS)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_ADDRS start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    std::vector<float*> addrs;

    lookupKeys = {0, 1, 2, 3, 4};
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs), H_OK);

    // lookupkeys 为空
    lookupKeys = {};
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs), H_OK);

    lookupKeys = {0};
    ASSERT_EQ(embCache->EmbeddingLookupAddrs("not_a_table", lookupKeys, addrs), H_TABLE_NOT_EXIST);

    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tooLongTableName, lookupKeys, addrs), H_TABLE_NAME_TOO_LONG);

    lookupKeys = {5};
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs), H_HOST_VOCAB_SIZE_TOO_SMALL);

    lookupKeys = {5};
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs, 1), H_HOST_VOCAB_SIZE_TOO_SMALL);

    lookupKeys = {0, 1, 4};
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs), H_OK);

    lookupKeys = {0, 1, 4};
    uint32_t threadNum = std::thread::hardware_concurrency();
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs, threadNum + 1), H_THREAD_NUM_ERROR);
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs, threadNum), H_OK);
    // 单线程lookup
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs, 1), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs, 0), H_THREAD_NUM_ERROR);
    embCache->Destroy();
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP_ADDRS_DATA)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_ADDRS_DATA start=============");
    factory->CreateEmbCacheManager(embCache);
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 3000000;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 39;
    uint32_t devVocabSize = 100000;
    EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::string normalInitializeName = "random_normal_initializer";
    std::string constantInitializeName = "constant_initializer";
    EmbCache::NormalInitializerInfo normalInitializerInfo(0, 0.05, 0, 1.0);
    EmbCache::ConstantInitializerInfo constantInitializerInfo(0.233, 1.0);

    std::string truncatedNormalInitializeName = "truncated_normal_initializer";
    // 加入所有初始化器的所有分支
    std::vector<EmbCache::InitializerInfo> initializeInfos = {
        EmbCache::InitializerInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo),
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, 0, normalInitializerInfo),
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize, constantInitializerInfo),
        EmbCache::InitializerInfo(constantInitializeName, 2 * embeddingSize, 0, constantInitializerInfo),
        EmbCache::InitializerInfo(truncatedNormalInitializeName, 2 * embeddingSize, embeddingSize,
                                  normalInitializerInfo),
        EmbCache::InitializerInfo(truncatedNormalInitializeName, 3 * embeddingSize, 0, normalInitializerInfo),
    };
    // 正确创建
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos), H_OK);
    std::vector<uint64_t> lookupKeys;
    std::vector<float*> addrs;
    lookupKeys = GenKeys(hostVocabSize, 123321);
    ASSERT_EQ(embCache->EmbeddingLookupAddrs(tableName, lookupKeys, addrs), H_OK);

    long double sum = 0.0;
    long double cnt = 0.0;
    long double accum = 0.0;
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        // normalInitializer 生成数据
        for (uint32_t j = 0; j < embeddingSize; j++) {
            sum += addrs[i][j];
            cnt++;
        }

        // constantInitializer 生成数据
        for (uint32_t j = embeddingSize; j < 2 * embeddingSize; j++) {
            ASSERT_LE(std::abs(addrs[i][j] - 0.233), 1e-6f);
        }
        // truncatedNormalInitializer 生成数据
        for (uint32_t j = 2 * embeddingSize; j < 3 * embeddingSize; j++) {
            // 在[-2*stddev, 2*stddev]范围中
            ASSERT_LE(std::abs(addrs[i][j]), 0.1f + 1e-6f);
        }
    }

    long double mean = sum / cnt;
    for (uint32_t i = 0; i < lookupKeys.size(); ++i) {
        for (uint32_t j = 0; j < embeddingSize; j++) {
            accum += (addrs[i][j] - mean) * (addrs[i][j] - mean);
        }
    }
    long double stdev = sqrt(accum / cnt);
    ASSERT_LE(std::abs(mean), 5e-6f);
    ASSERT_LE(std::abs(stdev - 0.05), 5e-6f);
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_ADDRS_DATA end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP_300W)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_300W start=============");
    factory->CreateEmbCacheManager(embCache);
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 3000000;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 39;
    uint32_t devVocabSize = 100000;
    EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::string normalInitializeName = "random_normal_initializer";
    std::string constantInitializeName = "constant_initializer";
    EmbCache::NormalInitializerInfo normalInitializerInfo(0, 0.05, 0, 1.0);
    EmbCache::ConstantInitializerInfo constantInitializerInfo(0.233, 1.0);

    std::string truncatedNormalInitializeName = "truncated_normal_initializer";
    // 加入所有初始化器的所有分支
    std::vector<EmbCache::InitializerInfo> initializeInfos = {
        EmbCache::InitializerInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo),
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, 0, normalInitializerInfo),
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize, constantInitializerInfo),
        EmbCache::InitializerInfo(constantInitializeName, 2 * embeddingSize, 0, constantInitializerInfo),
        EmbCache::InitializerInfo(truncatedNormalInitializeName, 2 * embeddingSize, embeddingSize,
                                  normalInitializerInfo),
        EmbCache::InitializerInfo(truncatedNormalInitializeName, 3 * embeddingSize, 0, normalInitializerInfo),
    };
    // 正确创建
    ASSERT_EQ(embCache->CreateCacheForTable(embCacheInfo, initializeInfos), H_OK);
    std::vector<uint64_t> lookupKeys;
    float* addr;
    lookupKeys = GenKeys(hostVocabSize, 123321);
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);

    long double sum = 0.0;
    long double cnt = 0.0;
    long double accum = 0.0;
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        // normalInitializer 生成数据
        for (uint32_t j = 0; j < embeddingSize; j++) {
            sum += addr[i * extEmbeddingSize + j];
            cnt++;
        }

        // constantInitializer 生成数据
        for (uint32_t j = embeddingSize; j < 2 * embeddingSize; j++) {
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j] - 0.233), 1e-6f);
        }
        // truncatedNormalInitializer 生成数据
        for (uint32_t j = 2 * embeddingSize; j < 3 * embeddingSize; j++) {
            // 在[-2*stddev, 2*stddev]范围中
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j]), 0.1f + 1e-6f);
        }
    }

    long double mean = sum / cnt;
    for (uint32_t i = 0; i < lookupKeys.size(); ++i) {
        for (uint32_t j = 0; j < embeddingSize; j++) {
            accum += (addr[i * extEmbeddingSize + j] - mean) * (addr[i * extEmbeddingSize + j] - mean);
        }
    }
    long double stdev = sqrt(accum / cnt);
    ASSERT_LE(std::abs(mean), 5e-6f);
    ASSERT_LE(std::abs(stdev - 0.05), 5e-6f);
    free(addr);
    CTRLog(CTRLogLevel::INFO, "===========GenerateData end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP_AND_REMOVE)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_AND_REMOVE start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    float* addr;

    lookupKeys = {0, 1, 2, 3, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr), H_OK);
    free(addr);

    // lookupkeys 为空
    lookupKeys = {};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr), H_OK);
    free(addr);

    lookupKeys = {0};
    addr = nullptr;
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove("not_a_table", lookupKeys, addr), H_TABLE_NOT_EXIST);

    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tooLongTableName, lookupKeys, addr), H_TABLE_NAME_TOO_LONG);

    lookupKeys = {0};
    addr = nullptr;
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr), H_ADDRESS_NULL);

    lookupKeys = {0, 1, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    uint32_t threadNum = std::thread::hardware_concurrency();
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr, threadNum + 1), H_THREAD_NUM_ERROR);
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr, threadNum), H_OK);
    // 单线程lookup
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr, 1), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr, 0), H_THREAD_NUM_ERROR);
    free(addr);
    embCache->Destroy();

    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_AND_REMOVE end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP_AND_REMOVE_2)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_AND_REMOVE_2 start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 200;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    float* addr;

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 2; j++) {
            lookupKeys.emplace_back(i);
        }
    }

    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr, 1), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr), H_OK);
    free(addr);
    embCache->Destroy();

    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_AND_REMOVE_2 end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    float* addr;

    lookupKeys = {0, 1, 2, 3, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    free(addr);

    // lookupkeys 为空
    lookupKeys = {};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    free(addr);

    lookupKeys = {0};
    addr = nullptr;
    ASSERT_EQ(embCache->EmbeddingLookup("not_a_table", lookupKeys, addr), H_TABLE_NOT_EXIST);

    ASSERT_EQ(embCache->EmbeddingLookup(tooLongTableName, lookupKeys, addr), H_TABLE_NAME_TOO_LONG);

    lookupKeys = {5};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_HOST_VOCAB_SIZE_TOO_SMALL);
    free(addr);

    lookupKeys = {0};
    addr = nullptr;
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_ADDRESS_NULL);

    lookupKeys = {0, 1, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    free(addr);

    lookupKeys = {0, 1, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    uint32_t threadNum = std::thread::hardware_concurrency();
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr, threadNum + 1), H_THREAD_NUM_ERROR);
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr, threadNum), H_OK);
    // 单线程lookup
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr, 1), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr, 0), H_THREAD_NUM_ERROR);
    free(addr);
    embCache->Destroy();

    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_LOOKUP_AND_REMOVE_300W)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_AND_REMOVE_300W start=============");
    std::string tableName = "test_table";
    std::vector<uint64_t> lookupKeys;
    float* newEmb;

    // 300w个key
    uint32_t hostVocabSize = 3000000;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 100000;
    embCache = ConstZeroCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    lookupKeys = GenUniqueKeys(hostVocabSize);
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            newEmb[i * extEmbeddingSize + j] = i + 0.01f * j;  // 生成特殊数据
        }
    }
    CTRLog(CTRLogLevel::INFO, "gen done");
    // 把特殊数据放到表中
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);
    CTRLog(CTRLogLevel::INFO, "EmbeddingUpdate done");

    float* addr;
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    // 查询特殊数据
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    CTRLog(CTRLogLevel::INFO, "EmbeddingLookup done");
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            // 验证表中数据正确性
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j] - (i + 0.01f * j)), 1e-6f);
        }
    }
    free(addr);
    addr = nullptr;

    // Remove之后再Lookup，观察这些embedding是不是被正确remove
    // 首先确认EmbeddingLookupAndRemove不会报错
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookupAndRemove(tableName, lookupKeys, addr, 4), H_OK);
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            // 验证表中数据正确性
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j] - (i + 0.01f * j)), 1e-6f);
        }
    }
    free(addr);
    addr = nullptr;
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    // 然后再lookup，并确保lookup不会报错
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    // 因为用const zero初始化， EmbeddingLookupAndRemove之后再lookup，结果应该全是0
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            // 验证表中数据正确性
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j] - 0), 1e-6f);
        }
    }
    free(addr);

    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_LOOKUP_AND_REMOVE_300W end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_UPDATE_300W)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_UPDATE_300W start=============");
    std::string tableName = "test_table";
    std::vector<uint64_t> lookupKeys;
    float* newEmb;

    // 300w个key
    uint32_t hostVocabSize = 3000000;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 100000;
    embCache = ConstZeroCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize, 50000, 6);
    lookupKeys = GenKeys(hostVocabSize, 123321);
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            newEmb[i * extEmbeddingSize + j] = i + 0.01f * j;  // 生成特殊数据
        }
    }
    CTRLog(CTRLogLevel::INFO, "gen done");
    // 把特殊数据放到表中
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);
    CTRLog(CTRLogLevel::INFO, "EmbeddingUpdate done");

    float* addr;
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    // 查询特殊数据
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    CTRLog(CTRLogLevel::INFO, "EmbeddingLookup done");
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            // 验证表中数据正确性
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j] - (i + 0.01f * j)), 1e-6f);
        }
    }
    // Remove之后再Lookup，观察这些embedding是不是被正确remove
    // 首先确认remove不会报错
    ASSERT_EQ(embCache->RemoveEmbsByKeys(tableName, lookupKeys), H_OK);
    // 然后再lookup，并确保lookup不会报错
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    // 因为用const zero初始化， 删除之后再lookup，结果应该全是0
    for (uint32_t i = 0; i < lookupKeys.size(); i++) {
        for (uint32_t j = 0; j < extEmbeddingSize; j++) {
            // 验证表中数据正确性
            ASSERT_LE(std::abs(addr[i * extEmbeddingSize + j] - 0), 1e-6f);
        }
    }
    free(addr);

    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_UPDATE_300W end=============");
}

TEST_F(EmbCacheTest, EMBEDDING_UPDATE)
{
    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_UPDATE start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    float* newEmb;

    lookupKeys = {0, 1, 2, 3, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }

    // 更新存在的table，应当正常更新
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    lookupKeys = {0};
    newEmb = nullptr;
    // 更新不存在的table
    ASSERT_EQ(embCache->EmbeddingUpdate("not_a_table", lookupKeys, newEmb), H_TABLE_NOT_EXIST);

    // 表名超过上限
    ASSERT_EQ(embCache->EmbeddingUpdate(tooLongTableName, lookupKeys, newEmb), H_TABLE_NAME_TOO_LONG);

    lookupKeys = {5};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }

    // 当前embLocalTable中存储的key已达到hostVocabSize上限，并继续添加新key
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_HOST_VOCAB_SIZE_TOO_SMALL);
    free(newEmb);

    lookupKeys = {0};
    newEmb = nullptr;
    // 传入embAddr为空指针
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_ADDRESS_NULL);

    // 更新存在于table的keys, 传入embAddr不为空指针
    lookupKeys = {0, 1, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    // 线程数未超过核数
    lookupKeys = {0, 1, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb, 4), H_OK);
    free(newEmb);

    // 线程数等于核数
    uint32_t processCoreNum = std::thread::hardware_concurrency();
    lookupKeys = {0, 1, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb, processCoreNum), H_OK);
    free(newEmb);

    // 线程数大于核数
    processCoreNum = std::thread::hardware_concurrency();
    lookupKeys = {0, 1, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb, processCoreNum + 1), H_THREAD_NUM_ERROR);
    free(newEmb);

    // 线程数为0
    processCoreNum = std::thread::hardware_concurrency();
    lookupKeys = {0, 1, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb, 0), H_THREAD_NUM_ERROR);
    free(newEmb);

    // 线程数为1
    lookupKeys = {0, 1, 4};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb, 1), H_OK);
    free(newEmb);

    // lookupkeys为空
    lookupKeys = {};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb, 1), H_OK);
    free(newEmb);

    TearDown();

    // 更新不存在于table的key，且当前embLocalTable中存储的key未达到hostVocabSize上限，继续添加新key
    tableName = "test_table_one";
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    lookupKeys = {0, 1};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb);
    free(newEmb);
    lookupKeys = {2, 3};
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    CTRLog(CTRLogLevel::INFO, "===========EMBEDDING_UPDATE end=============");
}

TEST_F(EmbCacheTest, GetSwapPairsAndKey2Offset)
{
    CTRLog(CTRLogLevel::INFO, "===========GetSwapPairsAndKey2Offset start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 100;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 10;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> insertKeys;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair, swapOutKoPair;

    // 使用不存在的table
    insertKeys = {0, 1, 2, 3, 4};
    std::vector<int64_t> paddingKeys = {1};
    EmbCache::EmbBaseInfo embBaseInfo(0, 0, "not_a_table", false, false, paddingKeys);
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair, swapOutKoPair),
              H_TABLE_NOT_EXIST);

    embBaseInfo.name = tooLongTableName;
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair, swapOutKoPair),
              H_TABLE_NAME_TOO_LONG);

    // 正常查找不存在的keys
    insertKeys = {0, 1, 2, 3, 4};
    embBaseInfo.name = tableName;
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair, swapOutKoPair), H_OK);
    bool ret1 = true;
    for (uint64_t i = 0; i < swapInKoPair.first.size(); i++) {
        if (swapInKoPair.first[i] != i) {
            string msg = "the " + std::to_string(i) + "th has key " + std::to_string(swapInKoPair.first[i]) +
                         ", but expect " + std::to_string(i);
            CTRLog(CTRLogLevel::INFO, msg.c_str());
            ret1 = false;
        }
    }
    ASSERT_EQ(ret1, true);

    // 正常查找存在的keys
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair2, swapOutKoPair2;
    insertKeys = {1, 2, 3};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair2, swapOutKoPair2), H_OK);
    uint64_t uint_zero = 0;
    ASSERT_EQ(swapInKoPair2.first.size(), uint_zero);

    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair3, swapOutKoPair3;
    insertKeys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // 使用非空的koPair
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair, swapOutKoPair3),
              H_ARG_NOT_EMPTY);
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair3, swapInKoPair),
              H_ARG_NOT_EMPTY);
    // 存入keys正好达到maxCacheSize上限值
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair3, swapOutKoPair3), H_OK);

    // 存入keys正好越过到maxCacheSize上限值
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair4, swapOutKoPair4;
    insertKeys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair4, swapOutKoPair4),
              H_MAX_CACHESIZE_TOO_SMALL);

    embCache->Destroy();
    // 单次存入keys超过maxCacheSize上限值
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair5, swapOutKoPair5;
    insertKeys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair5, swapOutKoPair5),
              H_MAX_CACHESIZE_TOO_SMALL);

    embCache->Destroy();
    // 单次存入keys正好达到上限值后，再次查找已存在的keys
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair6, swapOutKoPair6;
    insertKeys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair6, swapOutKoPair6), H_OK);

    embCache->Destroy();
    // 连续两次存入的keys未超过上限，第三次传入keys达到上限
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair7, swapOutKoPair7;
    insertKeys = {0, 1, 2, 3, 4};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair7, swapOutKoPair7), H_OK);

    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair8, swapOutKoPair8;
    insertKeys = {5, 6, 7, 8, 9};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair8, swapOutKoPair8), H_OK);

    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair9, swapOutKoPair9;
    insertKeys = {10, 11, 12, 13, 14};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair9, swapOutKoPair9), H_OK);

    embCache->Destroy();
    // 查询INVALID_KEY
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair10, swapOutKoPair10;
    uint64_t neg_one = -1;
    insertKeys = {neg_one, neg_one, neg_one, neg_one, neg_one};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair10, swapOutKoPair10), H_OK);
    ASSERT_EQ(swapInKoPair10.first.empty(), true);
    ASSERT_EQ(swapInKoPair10.second.empty(), true);
    ASSERT_EQ(swapOutKoPair10.first.empty(), true);
    ASSERT_EQ(swapOutKoPair10.second.empty(), true);

    // 查找空keys
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair11, swapOutKoPair11;
    insertKeys = {};
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, insertKeys, swapInKoPair11, swapOutKoPair11), H_OK);
    ASSERT_EQ(swapInKoPair11.first.empty(), true);
    ASSERT_EQ(swapInKoPair11.second.empty(), true);
    ASSERT_EQ(swapOutKoPair11.first.empty(), true);
    ASSERT_EQ(swapOutKoPair11.second.empty(), true);
    CTRLog(CTRLogLevel::INFO, "===========GetSwapPairsAndKey2Offset end=============");
}

bool checkKeys(std::set<uint64_t>& keySet, std::vector<std::set<uint64_t>>& historyKeyVec,
               const std::vector<uint64_t>& keys, const std::vector<uint64_t>& swapInKeys,
               const std::vector<uint64_t>& swapOutKeys, uint32_t maxCacheSize)
{
    std::set<uint64_t> newKeys;
    for (auto key : keys) {
        if (keySet.find(key) == keySet.end()) {
            newKeys.insert(key);
        }
        keySet.insert(key);
    }
    for (auto key : swapInKeys) {
        if (newKeys.find(key) == newKeys.end()) {
            CTRLog(CTRLogLevel::ERROR, "swapIn key error1");
            return false;
        }
    }
    if (swapInKeys.size() != newKeys.size()) {
        CTRLog(CTRLogLevel::ERROR, "swapIn key error2");
        return false;
    }
    historyKeyVec.insert(historyKeyVec.begin(), {keys.begin(), keys.end()});
    if (historyKeyVec.size() > 2) {
        historyKeyVec.pop_back();
    }
    for (auto key : swapOutKeys) {
        if (historyKeyVec[0].find(key) != historyKeyVec[0].end() ||
            historyKeyVec[1].find(key) != historyKeyVec[1].end()) {
            CTRLog(CTRLogLevel::ERROR, "swapOut key error1");
            return false;
        }
    }
    for (auto key : swapOutKeys) {
        if (keySet.find(key) == keySet.end()) {
            CTRLog(CTRLogLevel::ERROR, "swapOut key error2");
            return false;
        }
    }
    for (auto key : swapOutKeys) {
        keySet.erase(key);
    }
    if (keySet.size() > maxCacheSize) {
        CTRLog(CTRLogLevel::ERROR, "total key size error");
        return false;
    }
    return true;
}

bool checkOffsets(std::set<uint64_t>& offsetSet, const std::vector<uint64_t>& swapInOffsets,
                  const std::vector<uint64_t>& swapOutOffset)
{
    for (auto offset : swapOutOffset) {
        if (offsetSet.find(offset) == offsetSet.end()) {
            CTRLog(CTRLogLevel::ERROR, "swapOut offset error1");
            return false;
        }
    }

    for (auto offset : swapOutOffset) {
        offsetSet.erase(offset);
    }

    for (auto offset : swapInOffsets) {
        if (offsetSet.find(offset) != offsetSet.end()) {
            CTRLog(CTRLogLevel::ERROR, "swapIn offset error");
            return false;
        }
        offsetSet.insert(offset);
    }

    return true;
}

TEST_F(EmbCacheTest, DEVICE_COMBINE_TEST)
{
    CTRLog(CTRLogLevel::INFO, "===========DEVICE_COMBINE_TEST start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 4000000;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 30000;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::set<uint64_t> keySet;
    std::set<uint64_t> offsetSet;
    std::vector<std::set<uint64_t>> historyKeyVec;
    std::vector<std::vector<uint64_t>> historyOffsetVec;
    std::vector<uint64_t> lookupKeys;
    std::vector<uint64_t> check_keys;
    std::vector<int64_t> paddingKeys = {1};
    EmbCache::EmbBaseInfo embBaseInfo(0, 0, tableName, false, false, paddingKeys);
    for (uint32_t i = 0; i < 50; i++) {
        lookupKeys = GenKeys(10000, 123 + i, 0, 100000);
        check_keys = lookupKeys;
        std::pair<std::vector<uint64_t>, std::vector<uint64_t>> koPair1;
        std::pair<std::vector<uint64_t>, std::vector<uint64_t>> koPair2;
        ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, koPair1, koPair2), H_OK);
        bool retKey1 = checkKeys(keySet, historyKeyVec, check_keys, koPair1.first, koPair2.first, devVocabSize);
        bool retOffset1 = checkOffsets(offsetSet, koPair1.second, koPair2.second);
        ASSERT_EQ(retKey1, true);
        ASSERT_EQ(retOffset1, true);
    }

    CTRLog(CTRLogLevel::INFO, "===========DEVICE_COMBINE_TEST end=============");
}

TEST_F(EmbCacheTest, REMOVE_KEYS)
{
    CTRLog(CTRLogLevel::INFO, "===========REMOVE_KEYS start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 100;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 10;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    std::vector<uint64_t> removeKeys;
    float* addr;
    float* newEmb;

    for (uint32_t i = 0; i < hostVocabSize - 1; i++) {
        lookupKeys.emplace_back(i);
        for (uint32_t j = 0; j < hostVocabSize - 1; j++) {
            removeKeys.emplace_back(i + j);
        }
    }
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    free(addr);

    // 表存在
    ASSERT_EQ(embCache->RemoveEmbsByKeys(tableName, lookupKeys), H_OK);

    // 表不存在
    ASSERT_EQ(embCache->RemoveEmbsByKeys("not_a_table", lookupKeys), H_TABLE_NOT_EXIST);

    // 表名超过上限
    ASSERT_EQ(embCache->RemoveEmbsByKeys(tooLongTableName, lookupKeys), H_TABLE_NAME_TOO_LONG);

    // remove INVALID_KEY
    uint64_t neg_one = -1;
    lookupKeys = {neg_one, neg_one, neg_one, neg_one, neg_one};
    ASSERT_EQ(embCache->RemoveEmbsByKeys(tableName, lookupKeys), H_OK);

    // 判断embLocalTable是否remove掉记录信息
    lookupKeys = {0, 1, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    free(addr);

    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 999.99f;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    bool ret1 = true;
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        if (fabs(addr[i] - 999.99f) > 0.0000001) {
            ret1 = false;
        }
    }
    free(addr);
    ASSERT_EQ(ret1, true);

    ASSERT_EQ(embCache->RemoveEmbsByKeys(tableName, lookupKeys), H_OK);

    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    bool ret2 = true;
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        if (fabs(addr[i] - 999.99f) <= 0.0000001) {
            ret2 = false;
        }
    }
    free(addr);
    ASSERT_EQ(ret2, true);

    // 判断offsetMapper是否remove掉记录信息
    lookupKeys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair, swapOutKoPair;
    std::vector<int64_t> paddingKeys = {1};
    EmbCache::EmbBaseInfo embBaseInfo(0, 0, tableName, false, false, paddingKeys);
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, swapInKoPair, swapOutKoPair), H_OK);
    removeKeys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(embCache->RemoveEmbsByKeys(tableName, removeKeys), H_OK);
    std::vector<std::pair<uint64_t, uint64_t>> koVec;
    ASSERT_EQ(embCache->ExportDeviceKeyOffsetPairs(tableName, koVec), H_OK);
    bool ret3 = true;
    for (uint32_t i = 0; i < koVec.size(); i++) {
        if (std::find(removeKeys.begin(), removeKeys.end(), koVec[i].first) != removeKeys.end()) {
            ret3 = false;
        }
    }
    ASSERT_EQ(ret3, true);
    // 判断删除后，还能再添加
    lookupKeys = {9, 10, 11, 12, 13};
    std::vector<uint64_t> oldKeys = lookupKeys;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair2, swapOutKoPair2;
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, swapInKoPair2, swapOutKoPair2), H_OK);
    bool ret4 = true;
    for (uint32_t i = 0; i < 5; i++) {
        if (oldKeys[i] != swapInKoPair2.first[i]) {
            ret4 = false;
        }
    }
    bool ret5 = true;
    for (uint32_t i = 0; i < 5; i++) {
        if (lookupKeys[i] != swapInKoPair2.second[i]) {
            ret5 = false;
        }
    }
    ASSERT_EQ(ret4, true);
    ASSERT_EQ(ret5, true);
    ASSERT_EQ(swapInKoPair2.first.size(), 5ull);
    ASSERT_EQ(swapInKoPair2.second.size(), 5ull);
    ASSERT_EQ(swapOutKoPair2.first.empty(), true);
    ASSERT_EQ(swapOutKoPair2.second.empty(), true);

    removeKeys = {9, 10, 11, 3};
    ASSERT_EQ(embCache->RemoveEmbsByKeys(tableName, removeKeys), H_OK);
    std::vector<std::pair<uint64_t, uint64_t>> koVec2;
    ASSERT_EQ(embCache->ExportDeviceKeyOffsetPairs(tableName, koVec2), H_OK);
    bool ret6 = true;
    for (uint32_t i = 0; i < koVec2.size(); i++) {
        if (std::find(removeKeys.begin(), removeKeys.end(), koVec2[i].first) != removeKeys.end()) {
            ret6 = false;
        }
    }
    ASSERT_EQ(ret6, true);

    // 判断删除后，还能再添加
    lookupKeys = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<uint64_t> oldKeys2 = lookupKeys;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair3, swapOutKoPair3;
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, swapInKoPair3, swapOutKoPair3), H_OK);
    bool ret7 = true;
    for (uint32_t i = 0; i < 8; i++) {
        if (oldKeys2[i] != swapInKoPair3.first[i]) {
            ret7 = false;
        }
    }
    bool ret8 = true;
    for (uint32_t i = 0; i < 8; i++) {
        if (lookupKeys[i] != swapInKoPair3.second[i]) {
            ret8 = false;
        }
    }
    ASSERT_EQ(ret7, true);
    ASSERT_EQ(ret8, true);
    ASSERT_EQ(swapInKoPair3.first.size(), 8ull);
    ASSERT_EQ(swapInKoPair3.second.size(), 8ull);
    ASSERT_EQ(swapOutKoPair3.first.empty(), true);
    ASSERT_EQ(swapOutKoPair3.second.empty(), true);

    lookupKeys = {15};
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair4, swapOutKoPair4;
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, swapInKoPair4, swapOutKoPair4), H_OK);

    CTRLog(CTRLogLevel::INFO, "===========REMOVE_KEYS end=============");
}

TEST_F(EmbCacheTest, ExportDeviceKeyOffsetPairs)
{
    CTRLog(CTRLogLevel::INFO, "===========ExportDeviceKeyOffsetPairs start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 10;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 8;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    // 使用不存在的table名字
    std::vector<std::pair<uint64_t, uint64_t>> koVec;
    ASSERT_EQ(embCache->ExportDeviceKeyOffsetPairs("not_a_table", koVec), H_TABLE_NOT_EXIST);

    ASSERT_EQ(embCache->ExportDeviceKeyOffsetPairs(tooLongTableName, koVec), H_TABLE_NAME_TOO_LONG);

    // 正常export出koPair
    std::vector<uint64_t> lookupKeys;
    std::vector<uint64_t> checkKeys;
    lookupKeys = {6, 0, 8, 1, 3, 4};
    checkKeys = lookupKeys;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair, swapOutKoPair;
    std::vector<int64_t> paddingKeys = {1};
    EmbCache::EmbBaseInfo embBaseInfo(0, 0, tableName, false, false, paddingKeys);
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, swapInKoPair, swapOutKoPair), H_OK);
    std::vector<std::pair<uint64_t, uint64_t>> koVec2;
    ASSERT_EQ(embCache->ExportDeviceKeyOffsetPairs(tableName, koVec2), H_OK);
    ASSERT_EQ(koVec2.size(), lookupKeys.size());
    bool ret1 = true;
    for (uint32_t i = 0; i < koVec2.size(); i++) {
        if (koVec2[i].first != checkKeys[i] || koVec2[i].second != lookupKeys[i]) {
            ret1 = false;
        }
    }
    ASSERT_EQ(ret1, true);

    CTRLog(CTRLogLevel::INFO, "===========ExportDeviceKeyOffsetPairs end=============");
}

TEST_F(EmbCacheTest, GetEmbTableNames)
{
    CTRLog(CTRLogLevel::INFO, "===========GetEmbTableNames start=============");
    factory->CreateEmbCacheManager(embCache);
    uint32_t hostVocabSize = 10;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 8;
    std::vector<std::string> tableNameVec;
    tableNameVec.emplace_back("table1");
    tableNameVec.emplace_back("table2");
    tableNameVec.emplace_back("table3");
    for (const std::string tableName : tableNameVec) {
        EmbCache::EmbCacheInfo embCacheInfo(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

        EmbCache::NormalInitializerInfo normalInitializerInfo(0, 0.5, 0, 1.0);
        std::string normalInitializeName = "random_normal_initializer";
        EmbCache::InitializerInfo normalInitializeInfo(normalInitializeName, 0, embeddingSize, normalInitializerInfo);

        EmbCache::ConstantInitializerInfo constantInitializerInfo(0.233, 1.0);
        std::string constantInitializeName = "constant_initializer";
        EmbCache::InitializerInfo constantInitializeInfo(constantInitializeName, embeddingSize, embeddingSize,
                                                         constantInitializerInfo);

        std::vector<EmbCache::InitializerInfo> initializeInfos(extEmbeddingSize / embeddingSize);
        initializeInfos[0] = normalInitializeInfo;
        for (uint64_t i = 1; i < initializeInfos.size(); i++) {
            initializeInfos[i] = constantInitializeInfo;
        }
        embCache->CreateCacheForTable(embCacheInfo, initializeInfos, -1, hostVocabSize);
    }
    std::vector<std::string> allTableNames;
    std::vector<std::string> notEmptyVector = {"123"};
    ASSERT_EQ(embCache->GetEmbTableNames(notEmptyVector), H_ARG_NOT_EMPTY);

    ASSERT_EQ(embCache->GetEmbTableNames(allTableNames), H_OK);
    bool ret1 = true;
    for (auto tableName : allTableNames) {
        if (std::find(tableNameVec.begin(), tableNameVec.end(), tableName) == tableNameVec.end()) {
            ret1 = false;
        }
    }
    for (auto tableName : tableNameVec) {
        if (std::find(allTableNames.begin(), allTableNames.end(), tableName) == allTableNames.end()) {
            ret1 = false;
        }
    }
    ASSERT_EQ(ret1, true);

    CTRLog(CTRLogLevel::INFO, "===========GetEmbTableNames end=============");
}

TEST_F(EmbCacheTest, SERIALIZE)
{
    CTRLog(CTRLogLevel::INFO, "===========SERIALIZE start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    std::vector<uint64_t> lookupKeys;

    lookupKeys = {0};
    std::vector<char> buffer;
    ASSERT_EQ(embCache->Serialize("not_a_table", buffer), H_TABLE_NOT_EXIST);
    // 表名超过上限
    ASSERT_EQ(embCache->Serialize(tooLongTableName, buffer), H_TABLE_NAME_TOO_LONG);
    CTRLog(CTRLogLevel::INFO, "===========SERIALIZE end=============");
}

TEST_F(EmbCacheTest, DESERIALIZE)
{
    CTRLog(CTRLogLevel::INFO, "===========DESERIALIZE start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    std::vector<uint64_t> lookupKeys;

    lookupKeys = {0};
    std::vector<char> buffer = {'A', 'B', '1', '2'};
    ASSERT_EQ(embCache->Deserialize("not_a_table", buffer), H_TABLE_NOT_EXIST);

    ASSERT_EQ(embCache->Deserialize(tooLongTableName, buffer), H_TABLE_NAME_TOO_LONG);

    ASSERT_EQ(embCache->Deserialize(tableName, buffer), H_LOAD_ERROR);

    lookupKeys = {0, 1, 2, 3, 4};
    float* newEmb;
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);
    std::vector<char> buffer1;
    ASSERT_EQ(embCache->Serialize(tableName, buffer1), H_OK);
    buffer1.erase(buffer1.begin() + buffer1.size() / 2, buffer1.end());
    ASSERT_EQ(embCache->Deserialize(tableName, buffer1), H_LOAD_ERROR);

    CTRLog(CTRLogLevel::INFO, "===========DESERIALIZE end=============");
}

TEST_F(EmbCacheTest, SERIALIZE_DESERIALIZE)
{
    CTRLog(CTRLogLevel::INFO, "===========SERIALIZE_DESERIALIZE start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    std::vector<uint64_t> lookupKeys;
    lookupKeys = {0, 1, 2, 3, 4};
    float* newEmb;
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    std::vector<char> buffer1;
    std::vector<char> buffer2;

    ASSERT_EQ(embCache->Serialize(tableName, buffer1), H_OK);
    ASSERT_EQ(embCache->Deserialize(tableName, buffer1), H_OK);
    ASSERT_EQ(embCache->Serialize(tableName, buffer2), H_OK);
    ASSERT_EQ(buffer1.size(), buffer2.size());
    for (uint64_t i = 0; i < buffer1.size(); i++) {
        ASSERT_EQ(buffer1[i], buffer2[i]);
    }
    ASSERT_EQ(buffer1, buffer2);
    CTRLog(CTRLogLevel::INFO, "===========SERIALIZE_DESERIALIZE end=============");
}

TEST_F(EmbCacheTest, ERROR_INITIALIZER)
{
    CTRLog(CTRLogLevel::INFO, "===========ERROR_INITIALIZER start=============");
    uint32_t embeddingSize = 13;
    /* 对ConstantInitializerInfo的constValue和initK的校验 */
    std::string constantInitializeName = "constant_initializer";
    // 日志打印"constant value is less than -1000000000, and will use -1000000000."，并正常初始化InitializerInfo
    EmbCache::ConstantInitializerInfo constantInitializerInfo1(-1e9 - 1e8, 1.0);
    EmbCache::InitializerInfo constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize + 1, constantInitializerInfo1);

    // 日志打印"constant value is greater than 1000000000, and will use 1000000000."，并正常初始化InitializerInfo
    EmbCache::ConstantInitializerInfo constantInitializerInfo2(1e9 + 1e8, 1.0);
    constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize + 1, constantInitializerInfo2);

    // 日志打印"constant initK is greater than 10000, and will use 10000."，并正常初始化InitializerInfo
    EmbCache::ConstantInitializerInfo constantInitializerInfo3(0.233, 10001);
    constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize + 1, constantInitializerInfo3);

    // 日志打印"constant initK is less than -10000, and will use -10000."，并正常初始化InitializerInfo
    EmbCache::ConstantInitializerInfo constantInitializerInfo4(0.233, -10001);
    constantInitializeInfo =
        EmbCache::InitializerInfo(constantInitializeName, embeddingSize, embeddingSize + 1, constantInitializerInfo4);

    /* 对NormalIntializerInfo的mean、stdev和initK的校验 */
    std::string normalInitializeName = "random_normal_initializer";
    // 日志打印"random normal mean param is greater than 1000000000, and will use
    // 1000000000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo1(1e9 + 1e8, 0.05, 0, 1.0);
    EmbCache::InitializerInfo normalInitializeInfo =
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo1);

    // 日志打印"random normal mean param is less than -1000000000, and will use
    // -1000000000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo2(-1e9 - 1e8, 0.05, 0, 1.0);
    normalInitializeInfo =
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo2);

    // 日志打印"random normal stddev param is greater than 100, and will use 100."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo3(0, 101, 0, 1.0);
    normalInitializeInfo =
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo3);

    // 日志打印"random normal stddev param is less than 0, and will use 0."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo4(0, -1, 0, 1.0);
    normalInitializeInfo =
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo4);
    // 日志打印"random normal initK is greater than 10000, and will use 10000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo5(0, 0.05, 0, 10001);
    normalInitializeInfo =
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo5);
    // 日志打印"random normal initK is less than -10000, and will use -10000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo6(0, 0.05, 0, -10001);
    normalInitializeInfo =
        EmbCache::InitializerInfo(normalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo6);

    /* 对TruncatedNormalInitializer的mean、stdev以及initK的校验 */
    std::string truncatedNormalInitializeName = "truncated_normal_initializer";
    // 日志打印"truncated normal mean param is greater than 1000000000, and will use
    // 1000000000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo7(1e9 + 1e8, 0.05, 0, 1.0);
    EmbCache::InitializerInfo truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo7);

    // 日志打印"truncated normal mean param is less than -1000000000, and will use
    // -1000000000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo8(-1e9 - 1e8, 0.05, 0, 1.0);
    truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo8);

    // 日志打印"truncated normal stddev param is greater than 100, and will use 100."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo9(0, 101, 0, 1.0);
    truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo9);

    // 日志打印"truncated normal stddev param is less than 0.000000, and will use 0.000000."并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo10(0, -1, 0, 1.0);
    truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo10);
    // 日志打印"truncated normal initK is greater than 10000, and will use 10000."，并正常初始化InitializerInfo
    EmbCache::NormalInitializerInfo normalInitializerInfo11(0, 0.05, 0, 10001);
    truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo11);
    // 日志打印"truncated normal initK is less than -10000, and will use -10000."
    EmbCache::NormalInitializerInfo normalInitializerInfo12(0, 0.05, 0, -10001);
    truncatedNormalInitializeInfo =
        EmbCache::InitializerInfo(truncatedNormalInitializeName, embeddingSize, embeddingSize, normalInitializerInfo12);
    CTRLog(CTRLogLevel::INFO, "===========ERROR_INITIALIZER end=============");
}

TEST_F(EmbCacheTest, EmbeddingRemove)
{
    CTRLog(CTRLogLevel::INFO, "===========EmbeddingRemove start=============");
    std::string tableName = "test_table";
    uint32_t hostVocabSize = 100;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint32_t devVocabSize = 100;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys;
    std::vector<uint64_t> removeKeys;
    float* addr;
    float* newEmb;

    for (uint32_t i = 0; i < hostVocabSize - 1; i++) {
        lookupKeys.emplace_back(i);
        for (uint32_t j = 0; j < hostVocabSize - 1; j++) {
            removeKeys.emplace_back(i + j);
        }
    }
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    // 表存在
    ASSERT_EQ(embCache->EmbeddingRemove(tableName, lookupKeys), H_OK);
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    // 单线程
    ASSERT_EQ(embCache->EmbeddingRemove(tableName, lookupKeys, 1), H_OK);

    free(addr);
    // REMOVE空keys
    std::vector<uint64_t> emptyRemoveKeys;
    ASSERT_EQ(embCache->EmbeddingRemove(tableName, emptyRemoveKeys), H_OK);

    // 表不存在
    ASSERT_EQ(embCache->EmbeddingRemove("not_a_table", lookupKeys), H_TABLE_NOT_EXIST);
    // 表名超过上限
    ASSERT_EQ(embCache->EmbeddingRemove(tooLongTableName, lookupKeys), H_TABLE_NAME_TOO_LONG);

    // remove INVALID_KEY
    uint64_t neg_one = -1;
    lookupKeys = {neg_one, neg_one, neg_one, neg_one, neg_one};
    ASSERT_EQ(embCache->EmbeddingRemove(tableName, lookupKeys), H_OK);

    // 判断embLocalTable是否remove掉记录信息
    lookupKeys = {0, 1, 4};
    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    free(addr);

    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 999.99f;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    bool ret1 = true;
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        if (fabs(addr[i] - 999.99f) > 0.0000001) {
            ret1 = false;
        }
    }
    free(addr);
    ASSERT_EQ(ret1, true);

    ASSERT_EQ(embCache->EmbeddingRemove(tableName, lookupKeys), H_OK);

    addr = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    ASSERT_EQ(embCache->EmbeddingLookup(tableName, lookupKeys, addr), H_OK);
    bool ret2 = true;
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        if (fabs(addr[i] - 999.99f) <= 0.0000001) {
            ret2 = false;
        }
    }
    free(addr);
    ASSERT_EQ(ret2, true);

    // 判断offsetMapper是否remove掉记录信息
    lookupKeys = {6, 0, 8, 1, 3, 4};
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair, swapOutKoPair;
    std::vector<int64_t> paddingKeys = {1};
    EmbCache::EmbBaseInfo embBaseInfo(0, 0, tableName, false, false, paddingKeys);
    ASSERT_EQ(embCache->GetSwapPairsAndKey2Offset(embBaseInfo, lookupKeys, swapInKoPair, swapOutKoPair), H_OK);
    removeKeys = {0, 1, 4};
    ASSERT_EQ(embCache->EmbeddingRemove(tableName, removeKeys), H_OK);

    CTRLog(CTRLogLevel::INFO, "===========EmbeddingRemove end=============");
}

TEST_F(EmbCacheTest, GET_EMB_TABLE_INFO)
{
    CTRLog(CTRLogLevel::INFO, "===========GET_EMB_TABLE_INFO start=============");
    std::string tableName = "test_table";
    uint64_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint64_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    std::vector<uint64_t> lookupKeys;
    lookupKeys = {0, 1, 2, 3, 4};
    float* newEmb;
    newEmb = (float*)malloc(lookupKeys.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys.size() * extEmbeddingSize; i++) {
        newEmb[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys, newEmb), H_OK);
    free(newEmb);

    std::vector<uint64_t> keys;
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<float>> optimizerSlots;

    ASSERT_EQ(embCache->GetEmbTableInfos("Invalid_table_name", keys, embeddings, optimizerSlots), H_TABLE_NOT_EXIST);
    ASSERT_EQ(embCache->GetEmbTableInfos(tooLongTableName, keys, embeddings, optimizerSlots), H_TABLE_NAME_TOO_LONG);
    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys, embeddings, optimizerSlots), H_OK);
    bool ret = true;
    if (keys.size() != 5) {
        ret = false;
    }
    uint32_t optimizerSlotSize = extEmbeddingSize - embeddingSize;
    for (auto key : keys) {
        auto it = std::find(lookupKeys.begin(), lookupKeys.end(), key);
        if (it == lookupKeys.end()) {
            ret = false;
            break;
        }
        uint32_t index = it - lookupKeys.begin();
        for (uint32_t i = 0; i < embeddingSize; i++) {
            if (fabs(embeddings[index][i] - 0.01f * (i + index * extEmbeddingSize)) > 0.0000001) {
                ret = false;
            }
        }
        for (uint32_t i = 0; i < optimizerSlotSize; i++) {
            if (fabs(optimizerSlots[index][i] - 0.01f * (i + index * extEmbeddingSize + embeddingSize)) > 0.0000001) {
                ret = false;
            }
        }
    }
    ASSERT_EQ(ret, true);

    std::vector<uint64_t> keys2 = {1, 2, 3};
    std::vector<std::vector<float>> embeddings2;
    std::vector<std::vector<float>> optimizerSlots2;
    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys2, embeddings2, optimizerSlots2), H_ARG_NOT_EMPTY);

    std::vector<uint64_t> keys3;
    std::vector<std::vector<float>> embeddings3;
    std::vector<std::vector<float>> optimizerSlots3;
    embeddings3.emplace_back(std::vector<float>({0.1f, 0.2f}));
    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys3, embeddings3, optimizerSlots3), H_ARG_NOT_EMPTY);

    std::vector<uint64_t> keys4;
    std::vector<std::vector<float>> embeddings4;
    std::vector<std::vector<float>> optimizerSlots4;
    optimizerSlots4.emplace_back(std::vector<float>({0.1f, 0.2f}));
    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys4, embeddings4, optimizerSlots4), H_ARG_NOT_EMPTY);
    embCache->Destroy();

    hostVocabSize = 5;
    embeddingSize = 13;
    extEmbeddingSize = 13;
    devVocabSize = 2;

    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);
    std::vector<uint64_t> lookupKeys2;
    lookupKeys2 = {0, 1, 2, 3, 4};
    float* newEmb2;
    newEmb2 = (float*)malloc(lookupKeys2.size() * extEmbeddingSize * sizeof(float));
    for (uint32_t i = 0; i < lookupKeys2.size() * extEmbeddingSize; i++) {
        newEmb2[i] = 0.01f * i;
    }
    ASSERT_EQ(embCache->EmbeddingUpdate(tableName, lookupKeys2, newEmb2), H_OK);
    free(newEmb2);

    std::vector<uint64_t> keys5;
    std::vector<std::vector<float>> embeddings5;
    std::vector<std::vector<float>> optimizerSlots5;

    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys5, embeddings5, optimizerSlots5), H_OK);
    bool ret2 = true;
    if (keys.size() != 5) {
        ret2 = false;
    }
    for (auto key : keys) {
        auto it = std::find(lookupKeys2.begin(), lookupKeys2.end(), key);
        if (it == lookupKeys2.end()) {
            ret2 = false;
            break;
        }
        uint32_t index = it - lookupKeys2.begin();
        for (uint32_t i = 0; i < embeddingSize; i++) {
            if (fabs(embeddings5[index][i] - 0.01f * (i + index * extEmbeddingSize)) > 0.0000001) {
                ret2 = false;
            }
        }
    }
    if (!optimizerSlots5.empty()) {
        ret2 = false;
    }
    ASSERT_EQ(ret2, true);

    CTRLog(CTRLogLevel::INFO, "===========GET_EMB_TABLE_INFO end=============");
}

TEST_F(EmbCacheTest, LOAD_EMB_TABLE_INFO)
{
    CTRLog(CTRLogLevel::INFO, "===========LOAD_EMB_TABLE_INFO start=============");
    std::string tableName = "test_table";
    uint64_t hostVocabSize = 5;
    uint32_t embeddingSize = 13;
    uint32_t extEmbeddingSize = 26;
    uint64_t devVocabSize = 2;
    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    std::vector<uint64_t> keys;
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<float>> optimizerSlots;

    keys = {0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < keys.size(); i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings.emplace_back(curEmbedding);
    }
    uint32_t optimizerSlotSize = extEmbeddingSize - embeddingSize;
    for (uint64_t i = 0; i < keys.size(); i++) {
        std::vector<float> curOptimizerSlot;
        for (uint64_t j = 0; j < optimizerSlotSize; j++) {
            curOptimizerSlot.emplace_back(0.01f * (i * extEmbeddingSize + embeddingSize + j));
        }
        optimizerSlots.emplace_back(curOptimizerSlot);
    }
    ASSERT_EQ(embCache->LoadEmbTableInfos("Invalid_table_name", keys, embeddings, optimizerSlots), H_TABLE_NOT_EXIST);
    ASSERT_EQ(embCache->LoadEmbTableInfos(tooLongTableName, keys, embeddings, optimizerSlots), H_TABLE_NAME_TOO_LONG);
    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys, embeddings, optimizerSlots), H_OK);

    std::vector<uint64_t> keys2;
    std::vector<std::vector<float>> embeddings2;
    std::vector<std::vector<float>> optimizerSlots2;
    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys2, embeddings2, optimizerSlots2), H_OK);

    bool ret = true;
    if (keys2.size() != 5) {
        ret = false;
    }
    for (auto key : keys2) {
        auto it = std::find(keys.begin(), keys.end(), key);
        if (it == keys.end()) {
            ret = false;
            break;
        }
        uint32_t index = it - keys.begin();
        for (uint32_t i = 0; i < embeddingSize; i++) {
            if (fabs(embeddings2[index][i] - 0.01f * (i + index * extEmbeddingSize)) > 0.0000001) {
                ret = false;
            }
        }
        for (uint32_t i = 0; i < optimizerSlotSize; i++) {
            if (fabs(optimizerSlots2[index][i] - 0.01f * (i + index * extEmbeddingSize + embeddingSize)) > 0.0000001) {
                ret = false;
            }
        }
    }
    ASSERT_EQ(ret, true);

    std::vector<uint64_t> keys3;
    std::vector<std::vector<float>> embeddings3;
    std::vector<std::vector<float>> optimizerSlots3;

    keys3 = {0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < keys3.size() - 1; i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings3.emplace_back(curEmbedding);
    }
    for (uint64_t i = 0; i < keys3.size(); i++) {
        std::vector<float> curOptimizerSlot;
        for (uint64_t j = 0; j < optimizerSlotSize; j++) {
            curOptimizerSlot.emplace_back(0.01f * (i * extEmbeddingSize + embeddingSize + j));
        }
        optimizerSlots3.emplace_back(curOptimizerSlot);
    }
    // keys num != embeddings num
    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys3, embeddings3, optimizerSlots3), H_LOAD_ERROR);

    std::vector<uint64_t> keys4;
    std::vector<std::vector<float>> embeddings4;
    std::vector<std::vector<float>> optimizerSlots4;

    keys4 = {0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < keys4.size(); i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings4.emplace_back(curEmbedding);
    }
    for (uint64_t i = 0; i < keys4.size() - 1; i++) {
        std::vector<float> curOptimizerSlot;
        for (uint64_t j = 0; j < optimizerSlotSize; j++) {
            curOptimizerSlot.emplace_back(0.01f * (i * extEmbeddingSize + embeddingSize + j));
        }
        optimizerSlots4.emplace_back(curOptimizerSlot);
    }
    // keys num != optimizerSlots num
    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys4, embeddings4, optimizerSlots4), H_LOAD_ERROR);

    std::vector<uint64_t> keys5;
    std::vector<std::vector<float>> embeddings5;
    std::vector<std::vector<float>> optimizerSlots5;

    keys5 = {0, 1, 2, 3, 4, 5};
    for (uint64_t i = 0; i < keys5.size(); i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings5.emplace_back(curEmbedding);
    }
    for (uint64_t i = 0; i < keys5.size(); i++) {
        std::vector<float> curOptimizerSlot;
        for (uint64_t j = 0; j < optimizerSlotSize; j++) {
            curOptimizerSlot.emplace_back(0.01f * (i * extEmbeddingSize + embeddingSize + j));
        }
        optimizerSlots5.emplace_back(curOptimizerSlot);
    }
    // loadKeys num > hostVocabSize
    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys5, embeddings5, optimizerSlots5), H_LOAD_ERROR);

    std::vector<uint64_t> keys6;
    std::vector<std::vector<float>> embeddings6;
    std::vector<std::vector<float>> optimizerSlots6;

    keys6 = {0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < keys6.size(); i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize - 1; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings6.emplace_back(curEmbedding);
    }
    for (uint64_t i = 0; i < keys6.size(); i++) {
        std::vector<float> curOptimizerSlot;
        for (uint64_t j = 0; j < optimizerSlotSize; j++) {
            curOptimizerSlot.emplace_back(0.01f * (i * extEmbeddingSize + embeddingSize + j));
        }
        optimizerSlots6.emplace_back(curOptimizerSlot);
    }
    // entering embeddingSize != table embeddingSize
    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys6, embeddings6, optimizerSlots6), H_LOAD_ERROR);

    std::vector<uint64_t> keys7;
    std::vector<std::vector<float>> embeddings7;
    std::vector<std::vector<float>> optimizerSlots7;

    keys7 = {0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < keys7.size(); i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings7.emplace_back(curEmbedding);
    }
    for (uint64_t i = 0; i < keys7.size(); i++) {
        std::vector<float> curOptimizerSlot;
        for (uint64_t j = 0; j < optimizerSlotSize - 1; j++) {
            curOptimizerSlot.emplace_back(0.01f * (i * extEmbeddingSize + embeddingSize + j));
        }
        optimizerSlots7.emplace_back(curOptimizerSlot);
    }
    // entering optimizerSlotSize != table optimizerSlotSize
    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys7, embeddings7, optimizerSlots7), H_LOAD_ERROR);
    embCache->Destroy();

    hostVocabSize = 5;
    embeddingSize = 13;
    extEmbeddingSize = 13;
    devVocabSize = 2;

    embCache = SimpleCreateTable(tableName, hostVocabSize, embeddingSize, extEmbeddingSize, devVocabSize);

    std::vector<uint64_t> keys8;
    std::vector<std::vector<float>> embeddings8;
    std::vector<std::vector<float>> optimizerSlots8;

    keys8 = {0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < keys8.size(); i++) {
        std::vector<float> curEmbedding;
        for (uint64_t j = 0; j < embeddingSize; j++) {
            curEmbedding.emplace_back(0.01f * (i * extEmbeddingSize + j));
        }
        embeddings8.emplace_back(curEmbedding);
    }

    ASSERT_EQ(embCache->LoadEmbTableInfos(tableName, keys8, embeddings8, optimizerSlots8), H_OK);

    std::vector<uint64_t> keys9;
    std::vector<std::vector<float>> embeddings9;
    std::vector<std::vector<float>> optimizerSlots9;
    ASSERT_EQ(embCache->GetEmbTableInfos(tableName, keys9, embeddings9, optimizerSlots9), H_OK);

    double eps = 0.0000001;
    bool ret2 = true;
    if (keys9.size() != 5) {
        ret2 = false;
    }
    for (auto key : keys9) {
        auto it = std::find(keys9.begin(), keys9.end(), key);
        if (it == keys9.end()) {
            ret2 = false;
            break;
        }
        uint32_t index = it - keys9.begin();
        for (uint32_t i = 0; i < embeddingSize; i++) {
            if (fabs(embeddings9[index][i] - 0.01f * (i + index * extEmbeddingSize)) > eps) {
                ret2 = false;
            }
        }
    }
    if (!optimizerSlots9.empty()) {
        ret2 = false;
    }
    ASSERT_EQ(ret2, true);

    CTRLog(CTRLogLevel::INFO, "===========LOAD_EMB_TABLE_INFO end=============");
}
