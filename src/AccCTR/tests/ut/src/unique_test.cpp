/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

#include <sstream>
#include <fstream>
#include "unique_test.h"

FactoryPtr factory;

void UniqueTest::SetUpTestCase()
{
    Factory::Create(factory);
}

void UniqueTest::TearDownTestCase() {}

TEST_F(UniqueTest, Conf)
{
    std::cout << "===========Conf start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    ASSERT_EQ(Factory::Create(factory), -1); // 重复创建检查

    UniqueConf conf;
    conf.usePadding = true;
    conf.useSharding = true;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 1;
    conf.maxIdVal = 0;
    conf.shardingNum = 2;
    conf.paddingSize = 1;
    conf.outputType = OutputType::ENHANCED;
    ASSERT_EQ(unique->Initialize(conf), 4); // maxIdVal为0错误
    conf.maxIdVal = 9;
    conf.shardingNum = 0;
    ASSERT_EQ(unique->Initialize(conf), 4); // shardingNum为0错误
    conf.shardingNum = 4;
    conf.maxThreadNum = 0;
    ASSERT_EQ(unique->Initialize(conf), 1); // maxThreadNum为0错误
    conf.maxThreadNum = 2;
    conf.paddingSize = 0;
    ASSERT_EQ(unique->Initialize(conf), 4); // paddingSize为0错误
    conf.desiredSize = 0;
    ASSERT_EQ(unique->Initialize(conf), 4); // desiredSize为0错误
    conf.desiredSize = 1431655766;
    ASSERT_EQ(unique->Initialize(conf), 1); // desiredSize为1431655766错误
    conf.desiredSize = 6;
    conf.minThreadNum = 100;
    conf.maxThreadNum = 100;
    ASSERT_EQ(unique->Initialize(conf), 1); // minThreadNum过大错误
    conf.minThreadNum = 1;
    conf.maxThreadNum = 1;
    conf.paddingSize = 1;
    ASSERT_EQ(unique->Initialize(conf), 0); // 配置正确

    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(6);
    vector<uint32_t> index(6);
    int *idCnt = new int[6];
    int *idCntFill = new int[6];
    int *uniqueIdCntInBucket = new int[6];
    int64_t *uniqueIdInBucket = new int64_t[6];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = 6;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = nullptr;
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;
    uniqueOut.idCntFill = idCntFill;
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket;
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);
    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 3); // uniqueId 空指针
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.uniqueIdCntInBucket = nullptr;
    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 3); // uniqueIdCntInBucket空指针
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket;
    uniqueOut.idCntFill = nullptr;
    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 3); // idCntFill空指针
    uniqueOut.idCntFill = idCntFill;
    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 7); // padding长度过小
    std::cout << "===========Conf end=============" << std::endl;
}

// 使用了padding，但是没配sharding（padding依赖于sharding），报场景错误（9）
TEST_F(UniqueTest, usePaddingNoShardingErr)
{
    std::cout << "===========usePaddingNoShardingErr start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.usePadding = true;
    conf.useSharding = false;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.paddingSize = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 9);
    std::cout << "===========usePaddingNoShardingErr end=============" << std::endl;
}

TEST_F(UniqueTest, useNegativeDesiredSize)
{
    std::cout << "===========useNegativeDesiredSize start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = -1;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.outputType = OutputType::NORMAL;

    ASSERT_EQ(unique->Initialize(conf), 1);

    std::cout << "===========useNegativeDesiredSize end=============" << std::endl;
}

// DoUniqueNormal 只返回去重向量，恢复向量，长度
TEST_F(UniqueTest, DoUniqueNormal)
{
    std::cout << "===========DoUniqueNormal start=============" << std::endl;
    char *path = get_current_dir_name();
    std::string input_path(path);
    std::cout << "input_path:" + input_path + "/data30.txt" << std::endl;
    std::ifstream input(input_path + "/data30.txt");

    std::vector<int64_t> numbers;
    std::string line;
    while (std::getline(input, line, ',')) {
        std::istringstream in(line);
        std::copy(std::istream_iterator<int64_t>(in), std::istream_iterator<int64_t>(), std::back_inserter(numbers));
    }
    input.close();
    std::cout << "read data close, numbers size:" << numbers.size() << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.trace = true;
    conf.desiredSize = numbers.size();
    conf.dataType = DataType::INT64;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 8;
    conf.maxIdVal = 10000000000;
    conf.shardingNum = 1;
    conf.outputType = OutputType::NORMAL;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = numbers.size();
    vector<int64_t> inputId;
    for (size_t i = 0; i < numbers.size(); i++) {
        inputId.emplace_back(numbers[i]);
    }
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = nullptr;

    UniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.uniqueIdCnt = 0;
    ASSERT_EQ(unique->DoUnique(uniqueIn, uniqueOut), 3); // idCntFill空指针

    uniqueIn.inputId = inputId.data();
    unique->DoUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    for (uint32_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
    }

    unordered_set<int64_t> idsSet;
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        idsSet.insert(numbers[i]);
    }

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)idsSet.size());

    unique->UnInitialize();
    std::cout << "===========DoUniqueNormal end=============" << std::endl;
}

// 配Enhanced的conf，却使用normal接口
TEST_F(UniqueTest, UseErrOutputTypeEnhanced)
{
    std::cout << "===========UseErrOutputTypeEnhanced start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 1;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 1 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    UniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.uniqueIdCnt = 0;

    int ret = unique->DoUnique(uniqueIn, uniqueOut);
    ASSERT_EQ(ret, 8);

    unique->UnInitialize();
    std::cout << "===========UseErrOutputTypeEnhanced end=============" << std::endl;
}

// 配NORAML的conf，却使用extra接口
TEST_F(UniqueTest, UseErrOutputTypeNormal)
{
    std::cout << "===========UseErrOutputTypeNormal start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 1;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.outputType = OutputType::NORMAL;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();

    int ret = unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    ASSERT_EQ(ret, 8);

    unique->UnInitialize();
    std::cout << "===========UseErrOutputTypeNormal end=============" << std::endl;
}

// 用增强接口实现基础场景，只返回去重向量、恢复向量、长度
TEST_F(UniqueTest, DoEnhancedUnique)
{
    std::cout << "===========DoEnhancedUnique start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
    }

    unordered_set<int64_t> idsSet;
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        idsSet.insert(inputId[i]);
    }

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)idsSet.size());

    unique->UnInitialize();
    std::cout << "===========DoEnhancedUnique end=============" << std::endl;
}

// 开启特征计数但是不开启padding，且配置了idCntFill导致系统异常
TEST_F(UniqueTest, DoEnhancedUniqueErr)
{
    std::cout << "===========DoEnhancedUniqueErr start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.useIdCount = true;
    conf.useSharding = true;
    conf.outputType = OutputType::ENHANCED;


    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);

    int size = 6;
    vector<int> idCntFill(size);
    vector<int> uniqueIdCntInBucket(size);
    int64_t *uniqueIdInBucket = new int64_t[size];
    int *idCnt = new int[size];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();

    // 不开启padding counting但是传入地址
    uniqueOut.uniqueIdInBucket = uniqueIdInBucket;
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.idCntFill = idCntFill.data();
    uniqueOut.idCnt = idCnt;
    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
    }

    unordered_set<int64_t> idsSet;
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        idsSet.insert(inputId[i]);
    }

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)idsSet.size());

    unique->UnInitialize();
    std::cout << "===========DoEnhancedUniqueErr end=============" << std::endl;
}

// 用增强接口实现基础场景，padding 返回长度测试
TEST_F(UniqueTest, DoEnhancedUnique_UniqueIdSize)
{
    std::cout << "===========DoEnhancedUnique_UniqueIdSize start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
    }

    unordered_set<int64_t> idsSet;
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        idsSet.insert(inputId[i]);
    }

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)idsSet.size());

    unique->UnInitialize();
    std::cout << "===========DoEnhancedUnique_UniqueIdSize end=============" << std::endl;
}

// 增强接口配置特征计数，但特征计数向量设置为空
TEST_F(UniqueTest, idCntIsNull)
{
    std::cout << "===========idCntIsNull start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    int *idCnt = nullptr;

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;

    int ret = unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    ASSERT_EQ(ret, 3);

    unique->UnInitialize();
    std::cout << "===========idCntIsNull end=============" << std::endl;
}

// 增强接口配置padding,特征计数，但特征计数向量设置为空
TEST_F(UniqueTest, idCntIsNullSharding)
{
    std::cout << "===========idCntIsNullSharding start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useSharding = true;
    conf.usePadding = true;
    conf.useIdCount = true;
    conf.paddingSize = 5;
    conf.maxIdVal = 9;
    conf.shardingNum = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    int *idCnt = nullptr;
    int *idCntFill = nullptr;
    int *uniqueIdCntInBucket = new int[inputLen];
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;
    uniqueOut.idCntFill = idCntFill;
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket;
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    int ret = unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    ASSERT_EQ(ret, 3);

    unique->UnInitialize();
    std::cout << "===========idCntIsNullSharding end=============" << std::endl;
}

// 增强接口，配置sharding，特征计数
TEST_F(UniqueTest, DoUniqueShard)
{
    std::cout << "===========DoUniqueShard start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.useSharding = true;
    conf.useIdCount = true;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.paddingSize = 1;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> idCnt(inputLen);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);

    unordered_set<int> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        expectedIdCntMap[inputId[i]]++;
        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
        }
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

    vector<int> expectedIdCnt(uniqueOut.uniqueIdCnt);
    for (int i = 0; i < uniqueOut.uniqueIdCnt; i++) {
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }
    expectedIdCnt.resize(uniqueIn.inputIdCnt);

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    ASSERT_THAT(idCnt, testing::ElementsAreArray(expectedIdCnt));
    unique->UnInitialize();

    std::cout << "===========DoUniqueShard end=============" << std::endl;
}

// 增强接口，只配置sharding
TEST_F(UniqueTest, DoUniqueOnlyShard)
{
    std::cout << "===========DoUniqueOnlyShard start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.useSharding = true;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.paddingSize = 1;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);

    unordered_set<int> uniqueIdSet;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
        }
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    unique->UnInitialize();

    std::cout << "===========DoUniqueOnlyShard end=============" << std::endl;
}

// 增强接口，配置sharding，padding，特征计数
TEST_F(UniqueTest, DoUniquePadding)
{
    std::cout << "===========DoUniquePadding start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.usePadding = true;
    conf.useSharding = true;
    conf.paddingVal = -1;
    conf.paddingSize = 4;
    conf.desiredSize = 1; // 配置空间小于实际输入数组长度，验证正常运行
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 9;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 4, 4, 5, 6, 4, 0 };
    vector<int64_t> uniqueId(conf.paddingSize * conf.shardingNum);
    vector<uint32_t> index(inputLen);
    int *idCnt = new int[inputLen];
    vector<int> idCntFill(conf.paddingSize * conf.shardingNum);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = 6;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;
    uniqueOut.idCntFill = idCntFill.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);
    unordered_set<int> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        expectedIdCntMap[inputId[i]]++;

        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
        }
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, conf.shardingNum * conf.paddingSize);

    vector<int> expectedIdCnt(conf.paddingSize * conf.shardingNum);
    for (int i = 0; i < conf.paddingSize * conf.shardingNum; i++) {
        if (uniqueId[i] == -1) {
            continue;
        }
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    ASSERT_THAT(idCntFill, testing::ElementsAreArray(expectedIdCnt));
    ASSERT_EQ(uniqueOut.uniqueIdCnt, conf.paddingSize * conf.shardingNum);
    unique->UnInitialize();
    std::cout << "===========DoUniquePadding end=============" << std::endl;
}

// 增强接口，只配置特征计数，不配置线程池
TEST_F(UniqueTest, DoUniqueNoThreadPool)
{
    std::cout << "===========DoUniqueNoThreadPool start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 20; // 配置空间大于实际输入数组长度，验证正常运行
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 9;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> idCnt(inputLen);

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = 6;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt.data();

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
    }

    unordered_set<int64_t> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        uniqueIdSet.insert(inputId[i]);
        expectedIdCntMap[inputId[i]]++;
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

    vector<int> expectedIdCnt(uniqueOut.uniqueIdCnt);
    for (int i = 0; i < uniqueOut.uniqueIdCnt; i++) {
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }
    expectedIdCnt.resize(uniqueIn.inputIdCnt);

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(idCnt, testing::ElementsAreArray(expectedIdCnt));

    unique->UnInitialize();
    std::cout << "===========DoUniqueNoThreadPool end=============" << std::endl;
}

// 增强接口，配置sharding，特征计数，分桶数大于数据量
TEST_F(UniqueTest, DoUniqueShardNumberOversize)
{
    std::cout << "===========DoUniqueShardNumberOversize start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.useSharding = true;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 100;
    conf.shardingNum = 7;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> idCnt(inputLen);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);

    unordered_set<int> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        expectedIdCntMap[inputId[i]]++;
        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
        }
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

    int uniqueSum = 0;

    for (int i = 0; i < conf.shardingNum; i++) {
        uniqueSum += uniqueIdCntInBucket[i];
    }

    vector<int> expectedIdCnt(uniqueSum);
    for (int i = 0; i < uniqueSum; i++) {
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }
    expectedIdCnt.resize(uniqueIn.inputIdCnt);

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    ASSERT_THAT(idCnt, testing::ElementsAreArray(expectedIdCnt));
    unique->UnInitialize();

    std::cout << "===========DoUniqueShardNumberOversize end=============" << std::endl;
}

TEST_F(UniqueTest, DoUniqueSpecial)
{
    std::cout << "===========DoUniqueSpecial start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    int count = 1000000;
    UniqueConf conf;
    conf.paddingVal = -1;
    conf.usePadding = false;
    conf.useSharding = true;
    conf.desiredSize = 100;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 8;
    conf.maxIdVal = 0;
    conf.paddingSize = 4;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 4); // maxIdVal为0错误
    conf.maxIdVal = count + 1;
    ASSERT_EQ(unique->Initialize(conf), 0);

    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    std::vector<int64_t> inputData;
    inputData.resize(count);
    for (int i = 0; i < count; i++) {
        inputData[i] = count;
    }
    int64_t *uniqueData = new int64_t[count];
    uint32_t *index = new uint32_t[count];
    int *idCnt = new int[count];
    int *idCntFill = new int[count];
    int *uniqueIdCntInBucket = new int[count];
    int64_t *uniqueIdInBucket = new int64_t[count];
    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = count;
    uniqueIn.inputId = inputData.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueData);
    uniqueOut.index = index;
    uniqueOut.idCnt = idCnt;
    uniqueOut.idCntFill = idCntFill;
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket;
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    for (int i = 0; i < 2; i++) {
        ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 0);

        vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
        for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
            restoreIds[i] = uniqueData[index[i]];
        }

        ASSERT_THAT(inputData, testing::ElementsAreArray(restoreIds));
    }

    unique->UnInitialize();

    std::cout << "===========DoUniqueSpecial end=============" << std::endl;
}

// 增强接口，id数过大
TEST_F(UniqueTest, IdLarge)
{
    std::cout << "===========IdLarge start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 1;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    int *idCnt = new int[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;

    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 6); // ID太大
    std::cout << "===========IdLarge end=============" << std::endl;
}

// 增强接口，配置输入数据为int32，配置sharding，特征计数。
TEST_F(UniqueTest, DoUniqueNormalInt32)
{
    std::cout << "===========DoUniqueNormalInt32 start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.useSharding = true;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT32;
    conf.useIdCount = true;
    conf.maxIdVal = 9;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int32_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int32_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> idCnt(inputLen);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int32_t *uniqueIdInBucket = new int32_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    unordered_set<int> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    vector<int32_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        expectedIdCntMap[inputId[i]]++;
        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
        }
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

    vector<int> expectedIdCnt(uniqueOut.uniqueIdCnt);
    for (int i = 0; i < uniqueOut.uniqueIdCnt; i++) {
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }
    expectedIdCnt.resize(uniqueIn.inputIdCnt);

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    ASSERT_THAT(idCnt, testing::ElementsAreArray(expectedIdCnt));

    unique->UnInitialize();
    std::cout << "===========DoUniqueNormalInt32 end=============" << std::endl;
}

TEST_F(UniqueTest, DoUniqueMultipleTimes)
{
    std::cout << "===========DoUniqueMultipleTimes start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.maxIdVal = 9;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);


    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();

    for (int i = 0; i < 1000; i++) {
        unique->DoEnhancedUnique(uniqueIn, uniqueOut);

        vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
        for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
            restoreIds[i] = uniqueId[index[i]];
        }

        unordered_set<int64_t> idsSet;
        for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
            idsSet.insert(inputId[i]);
        }

        ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
        ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)idsSet.size());
    }

    unique->UnInitialize();
    std::cout << "===========DoUniqueMultipleTimes end=============" << std::endl;
}

TEST_F(UniqueTest, DoUniqueShardMultipleTimes)
{
    std::cout << "===========DoUniqueShardMultipleTimes start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.useSharding = true;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 9;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;

    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> idCnt(inputLen);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    for (int i = 0; i < 1000; i++) {
        unique->DoEnhancedUnique(uniqueIn, uniqueOut);

        vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
        vector<int> expectedUniqueIdCnt(conf.shardingNum);

        unordered_set<int> uniqueIdSet;
        map<int64_t, int> expectedIdCntMap;

        for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
            restoreIds[i] = uniqueId[index[i]];
            expectedIdCntMap[inputId[i]]++;
            if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
                continue;
            } else {
                uniqueIdSet.insert(inputId[i]);
                expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
            }
        }

        ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

        int uniqueSum = 0;

        for (int i = 0; i < conf.shardingNum; i++) {
            uniqueSum += uniqueIdCntInBucket[i];
        }

        vector<int> expectedIdCnt(uniqueSum);
        for (int i = 0; i < uniqueSum; i++) {
            expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
        }
        expectedIdCnt.resize(uniqueIn.inputIdCnt);

        ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
        ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
        ASSERT_THAT(idCnt, testing::ElementsAreArray(expectedIdCnt));
    }
    unique->UnInitialize();

    std::cout << "===========DoUniqueShardMultipleTimes end=============" << std::endl;
}

TEST_F(UniqueTest, DoUniquePaddingMultipleTimes)
{
    std::cout << "===========DoUniquePaddingMultipleTimes start=============" << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.usePadding = true;
    conf.useSharding = true;
    conf.paddingVal = -1;
    conf.paddingSize = 4;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 1;
    conf.maxIdVal = 9;
    conf.shardingNum = 2;
    conf.outputType = OutputType::ENHANCED;
    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 4, 4, 5, 6, 4, 0 };
    vector<int64_t> uniqueId(conf.paddingSize * conf.shardingNum);
    vector<uint32_t> index(inputLen);
    int *idCnt = new int[inputLen];
    vector<int> idCntFill(conf.paddingSize * conf.shardingNum);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = 6;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;
    uniqueOut.idCntFill = idCntFill.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    for (int i = 0; i < 1000; i++) {
        unique->DoEnhancedUnique(uniqueIn, uniqueOut);

        vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
        vector<int> expectedUniqueIdCnt(conf.shardingNum);
        unordered_set<int> uniqueIdSet;
        map<int64_t, int> expectedIdCntMap;

        for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
            restoreIds[i] = uniqueId[index[i]];
            expectedIdCntMap[inputId[i]]++;

            if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
                continue;
            } else {
                uniqueIdSet.insert(inputId[i]);
                expectedUniqueIdCnt[inputId[i] % conf.shardingNum]++;
            }
        }

        ASSERT_EQ(uniqueOut.uniqueIdCnt, conf.shardingNum * conf.paddingSize);

        vector<int> expectedIdCnt(conf.paddingSize * conf.shardingNum);
        for (int i = 0; i < conf.paddingSize * conf.shardingNum; i++) {
            if (uniqueId[i] == -1) {
                continue;
            }
            expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
        }

        ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
        ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
        ASSERT_THAT(idCntFill, testing::ElementsAreArray(expectedIdCnt));
    }

    unique->UnInitialize();
    std::cout << "===========DoUniquePaddingMultipleTimes end=============" << std::endl;
}

TEST_F(UniqueTest, IdCntSmall)
{
    std::cout << "===========IdCntSmall start=============" << std::endl;
    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    UniqueConf conf;
    conf.desiredSize = 6;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.maxIdVal = 100;
    conf.outputType = OutputType::ENHANCED;
    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    int inputLen = 6;
    vector<int64_t> inputId = { 5, 5, 3, 1, 5, 2 };
    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    int *idCnt = new int[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = 0;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;

    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 4); // idcnt过小
    std::cout << "===========IdCntSmall end=============" << std::endl;
}

TEST_F(UniqueTest, DoUniqueLotsDataFunction)
{
    std::cout << "===========DoUniqueLotsDataFunction start=============" << std::endl;
    char *path = get_current_dir_name();
    std::string input_path(path);
    std::cout << "input_path:" + input_path + "/data40.txt" << std::endl;
    std::ifstream input(input_path + "/data40.txt");

    std::vector<int64_t> numbers;
    std::string line;
    while (std::getline(input, line, ',')) {
        std::istringstream in(line);
        std::copy(std::istream_iterator<int64_t>(in), std::istream_iterator<int64_t>(), std::back_inserter(numbers));
    }
    input.close();
    std::cout << "read data close, numbers size:" << numbers.size() << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    int inputLen = numbers.size();
    UniqueConf conf;
    conf.useSharding = true;
    conf.desiredSize = 1; // 配置空间小于实际输入数组长度，验证正常运行
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 8;
    conf.maxIdVal = 10000000000;
    conf.shardingNum = 8;
    conf.outputType = OutputType::ENHANCED;
    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    vector<int64_t> inputId;

    for (size_t i = 0; i < numbers.size(); i++) {
        inputId.emplace_back(numbers[i]);
    }

    vector<int64_t> uniqueId(inputLen);
    vector<uint32_t> index(inputLen);
    vector<int> idCnt(inputLen);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = uniqueIdInBucket;

    unique->DoEnhancedUnique(uniqueIn, uniqueOut);

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);
    unordered_set<int64_t> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        expectedIdCntMap[inputId[i]]++;

        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[(uint64_t)(inputId[i]) % conf.shardingNum]++;
        }
    }

    ASSERT_EQ(uniqueOut.uniqueIdCnt, (int)uniqueIdSet.size());

    int uniqueSum = 0;

    for (int i = 0; i < conf.shardingNum; i++) {
        uniqueSum += uniqueIdCntInBucket[i];
    }

    vector<int> expectedIdCnt(uniqueSum);
    for (int i = 0; i < uniqueSum; i++) {
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }
    expectedIdCnt.resize(uniqueIn.inputIdCnt);

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    ASSERT_THAT(idCnt, testing::ElementsAreArray(expectedIdCnt));

    unique->UnInitialize();
    if (path) {
        free(path);
    }
    std::cout << "===========DoUniqueLotsDataFunction end=============" << std::endl;
}


TEST_F(UniqueTest, DoUniqueLotsDataPaddingFunction)
{
    std::cout << "===========DoUniqueLotsDataPaddingFunction start=============" << std::endl;
    char *path = get_current_dir_name();
    std::string input_path(path);
    std::cout << "input_path:" + input_path + "/data30.txt" << std::endl;
    std::ifstream input(input_path + "/data30.txt");

    std::vector<int64_t> numbers;
    std::string line;
    while (std::getline(input, line, ',')) {
        std::istringstream in(line);
        std::copy(std::istream_iterator<int64_t>(in), std::istream_iterator<int64_t>(), std::back_inserter(numbers));
    }
    input.close();
    std::cout << "read data close, numbers size:" << numbers.size() << std::endl;

    UniquePtr unique;
    ASSERT_EQ(factory->CreateUnique(unique), 0);

    int inputLen = numbers.size();
    UniqueConf conf;
    conf.trace = true;
    conf.usePadding = true;
    conf.useSharding = true;
    conf.paddingVal = -1;
    conf.paddingSize = 150000;
    conf.desiredSize = inputLen;
    conf.dataType = DataType::INT64;
    conf.useIdCount = true;
    conf.minThreadNum = 1;
    conf.maxThreadNum = 8;
    conf.maxIdVal = 10000000000;
    conf.shardingNum = 8;
    conf.outputType = OutputType::ENHANCED;
    ASSERT_EQ(unique->Initialize(conf), 0);
    unique->SetExternalThreadFuncInner(SimpleThreadPool::SyncRun);

    vector<int64_t> inputId;

    for (size_t i = 0; i < numbers.size(); i++) {
        inputId.emplace_back(numbers[i]);
    }

    vector<int64_t> uniqueId(conf.paddingSize * conf.shardingNum);
    vector<uint32_t> index(inputLen);
    int *idCnt = new int[inputLen];
    vector<int> idCntFill(conf.paddingSize * conf.shardingNum);
    vector<int> uniqueIdCntInBucket(conf.shardingNum);
    int64_t *uniqueIdInBucket = new int64_t[inputLen];

    UniqueIn uniqueIn;
    uniqueIn.inputIdCnt = inputLen;
    uniqueIn.inputId = inputId.data();

    EnhancedUniqueOut uniqueOut;
    uniqueOut.uniqueId = reinterpret_cast<void *>(uniqueId.data());
    uniqueOut.index = index.data();
    uniqueOut.idCnt = idCnt;
    uniqueOut.idCntFill = idCntFill.data();
    uniqueOut.uniqueIdCntInBucket = uniqueIdCntInBucket.data();
    uniqueOut.uniqueIdInBucket = reinterpret_cast<void *>(uniqueIdInBucket);

    for (int i = 0; i < 3; i++) {
        unique->DoEnhancedUnique(uniqueIn, uniqueOut);
    }

    vector<int64_t> restoreIds(uniqueIn.inputIdCnt);
    vector<int> expectedUniqueIdCnt(conf.shardingNum);
    unordered_set<int64_t> uniqueIdSet;
    map<int64_t, int> expectedIdCntMap;

    for (size_t i = 0; i < uniqueIn.inputIdCnt; i++) {
        restoreIds[i] = uniqueId[index[i]];
        expectedIdCntMap[inputId[i]]++;

        if (uniqueIdSet.find(inputId[i]) != uniqueIdSet.end()) {
            continue;
        } else {
            uniqueIdSet.insert(inputId[i]);
            expectedUniqueIdCnt[(uint64_t)(inputId[i]) % conf.shardingNum]++;
        }
    }

    vector<int> expectedIdCnt(conf.paddingSize * conf.shardingNum);
    for (int i = 0; i < conf.paddingSize * conf.shardingNum; i++) {
        if (uniqueId[i] == -1) {
            continue;
        }
        expectedIdCnt[i] = expectedIdCntMap[uniqueId[i]];
    }

    ASSERT_THAT(inputId, testing::ElementsAreArray(restoreIds));
    ASSERT_THAT(uniqueIdCntInBucket, testing::ElementsAreArray(expectedUniqueIdCnt));
    ASSERT_THAT(idCntFill, testing::ElementsAreArray(expectedIdCnt));

    unique->UnInitialize();
    ASSERT_EQ(unique->DoEnhancedUnique(uniqueIn, uniqueOut), 11);
    if (path) {
        free(path);
    }
    std::cout << "===========DoUniqueLotsDataPaddingFunction end=============" << std::endl;
}