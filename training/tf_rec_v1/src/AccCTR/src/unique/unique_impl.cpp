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

#include <thread>
#include "common/util/time_cost.h"
#include "unique_impl.h"

namespace ock {
namespace ctr {
UniqueImpl::UniqueImpl() {}

int UniqueImpl::Initialize(const UniqueConf &conf)
{
    uniqueConf = conf;
    int ret = CheckConf(uniqueConf);
    if (ret != H_OK) {
        return ret;
    }

    if ((uniqueConf.outputType == OutputType::NORMAL) || (!uniqueConf.useSharding)) {
        uniqueConf.shardingNum = 1; // 默认1个桶
    }

    GroupMethod groupMethod {};
    groupMethod.SetGroupCount(uniqueConf.shardingNum);
    groupMethod.SetStrategyFunByConf(uniqueConf.bucketStrategy);
    try {
        UnInitialize();
        unique = new ShardedDedup(groupMethod, uniqueConf);
        if (unique == nullptr) {
            return H_ADDRESS_NULL;
        }
    } catch (AllocError &) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "memory alloc error");
        return H_MEMORY_ALLOC_ERROR;
    } catch (NullptrError &) {
        return H_ADDRESS_NULL;
    }

    return H_OK;
}

int UniqueImpl::DoUnique(UniqueIn &uniqueIn, UniqueOut &uniqueOut)
{
    TimeCost doUniqueTimeCost;
    if (!IsInitialized()) {
        return H_UNIQUE_UNINITIALIZED_ERROR;
    }

    if (uniqueConf.outputType != OutputType::NORMAL) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "output type error, should be NORMAL");
        return H_OUTPUT_TYPE_ERROR;
    }

    int ret = CheckInput(uniqueIn, uniqueOut);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "input or conf is error");
        return ret;
    }

    UniqueOutSelf uniqueOutSelf;
    uniqueOutSelf.uniqueId = uniqueOut.uniqueId;
    uniqueOutSelf.index = uniqueOut.index;
    uniqueOutSelf.uniqueIdCnt = uniqueOut.uniqueIdCnt;

    if (uniqueConf.dataType == DataType::INT64) {
        ret = unique->Compute<DataType::INT64>(uniqueIn, uniqueOutSelf);
    } else if (uniqueConf.dataType == DataType::INT32) {
        ret = unique->Compute<DataType::INT32>(uniqueIn, uniqueOutSelf);
    }

    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "do unique error");
        return ret;
    }
    uniqueOut.uniqueIdCnt = uniqueOutSelf.uniqueIdCnt;

    std::stringstream sm;
    sm << "input id count " << uniqueIn.inputIdCnt << "; id count after unique " << uniqueOut.uniqueIdCnt <<
        "; unique id time cost " << doUniqueTimeCost.ElapsedMS() << " (ms)";
    ExternalLogger::PrintLog(LogLevel::INFO, sm.str(), uniqueConf.trace);

    // 资源回收
    return ret;
}

int UniqueImpl::DoEnhancedUnique(UniqueIn &uniqueIn, EnhancedUniqueOut &uniqueOut)
{
    TimeCost doEnhancedUniqueTimeCost;
    if (!IsInitialized()) {
        return H_UNIQUE_UNINITIALIZED_ERROR;
    }

    if (uniqueConf.outputType != OutputType::ENHANCED) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "output type error, should be ENHANCED");
        return H_OUTPUT_TYPE_ERROR;
    }

    int ret = CheckInput(uniqueIn, uniqueOut);
    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "input or conf is error.");
        return ret;
    }

    UniqueOutSelf uniqueOutSelf;
    uniqueOutSelf.uniqueId = uniqueOut.uniqueId;
    uniqueOutSelf.index = uniqueOut.index;
    uniqueOutSelf.uniqueIdCntInBucket = uniqueOut.uniqueIdCntInBucket;
    uniqueOutSelf.uniqueIdInBucket = uniqueOut.uniqueIdInBucket;
    if (uniqueConf.useIdCount) {
        uniqueOutSelf.idCnt = uniqueOut.idCnt;
        uniqueOutSelf.idCntFill = uniqueOut.idCntFill;
    }

    if (uniqueConf.dataType == DataType::INT64) {
        ret = unique->Compute<DataType::INT64>(uniqueIn, uniqueOutSelf);
    } else if (uniqueConf.dataType == DataType::INT32) {
        ret = unique->Compute<DataType::INT32>(uniqueIn, uniqueOutSelf);
    }

    if (ret != H_OK) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "do unique error");
        return ret;
    }

    if (uniqueConf.usePadding) {
        uniqueOut.uniqueIdCnt = uniqueConf.paddingSize * uniqueConf.shardingNum;
    } else {
        uniqueOut.uniqueIdCnt = uniqueOutSelf.uniqueIdCnt;
    }

    std::stringstream sm;
    sm << "input id count " << uniqueIn.inputIdCnt << "; id count after unique " << uniqueOutSelf.uniqueIdCnt <<
        "; unique id time cost " << doEnhancedUniqueTimeCost.ElapsedMS() << " (ms)";
    ExternalLogger::PrintLog(LogLevel::INFO, sm.str(), uniqueConf.trace);
    // 资源回收
    return ret;
}

void UniqueImpl::UnInitialize()
{
    if (unique != nullptr) {
        delete unique;
        unique = nullptr;
    }
}

int UniqueImpl::SetExternalThreadFuncInner(ExternalThread threadFunc)
{
    auto threader = ExternalThreader::Instance();
    if (threader == nullptr) {
        std::cout << "Failed to create threader instance" << std::endl;
        return H_NEW_OBJECT_FAILED;
    }

    threader->SetExternalLogFunction(threadFunc);
    return H_OK;
}

int UniqueImpl::CheckConf(const UniqueConf &conf)
{
    int ret = CheckNormalConf(conf);
    if (ret != H_OK) {
        return ret;
    }

    if (conf.outputType == OutputType::ENHANCED) {
        ret = CheckEnhancedUniqueConf(conf);
        if (ret != H_OK) {
            return ret;
        }
    }

    return H_OK;
}

int UniqueImpl::CheckNormalConf(const UniqueConf &conf)
{
    if (CheckInputZero(conf.maxIdVal, "maxIdVal") || CheckInputZero(conf.desiredSize, "desiredSize")) {
        return H_NUM_SMALL;
    }
    uint32_t processCoreNum = std::thread::hardware_concurrency();
    if (conf.maxThreadNum > processCoreNum) {
        std::stringstream sm;
        sm << "maxThreadNum can not larger than " << processCoreNum;
        ExternalLogger::PrintLog(LogLevel::ERROR, sm.str());
        return H_ERROR;
    }

    if (conf.maxThreadNum == 0 || conf.minThreadNum == 0 || conf.minThreadNum > conf.maxThreadNum) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "please check minThreadNum and maxThreadNum");
        return H_ERROR;
    }

    if (conf.desiredSize > MAX_DESIRED_SIZE) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "desiredSize can not larger than" + std::to_string(MAX_DESIRED_SIZE));
        return H_ERROR;
    }

    return H_OK;
}

int UniqueImpl::CheckEnhancedUniqueConf(const UniqueConf &conf)
{
    if (conf.usePadding) {
        if (!conf.useSharding) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "sharding is not enable."); // 使能padding时，先使能sharding
            return H_SCENE_ERROR;
        }

        if (CheckInputZero(conf.paddingSize, "paddingSize")) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "if usePadding is true, paddingSize can not be zero");
            return H_NUM_SMALL;
        }
    }

    if (conf.useSharding) {
        if (CheckInputZero(conf.shardingNum, "shardingNum")) {
            return H_NUM_SMALL;
        }
        if (conf.performance) {
            // 判断是不是2的N次幂
            bool isExponentOfTwo =
                (conf.shardingNum > 0) &&
                ((static_cast<unsigned int>(conf.shardingNum) & static_cast<unsigned int>(conf.shardingNum - 1)) == 0);
            if (!isExponentOfTwo) {
                ExternalLogger::PrintLog(LogLevel::ERROR, "if performance is true, shardingNum must be 2^N");
                return H_ERROR;
            }
        }
    }

    return H_OK;
}

int UniqueImpl::CheckInput(UniqueIn &uniqueIn, EnhancedUniqueOut &uniqueOut)
{
    if (CheckInputNull(uniqueIn.inputId, "inputId") || CheckInputNull(uniqueOut.uniqueId, "uniqueId") ||
        CheckInputNull(uniqueOut.index, "index")) {
        return H_ADDRESS_NULL;
    }

    if (uniqueConf.useSharding) {
        if (CheckInputNull(uniqueOut.uniqueIdInBucket, "uniqueIdInBucket") ||
            CheckInputNull(uniqueOut.uniqueIdCntInBucket, "uniqueIdCntInBucket")) {
            return H_ADDRESS_NULL;
        }
    }

    if (uniqueConf.useIdCount) {
        if (CheckInputNull(uniqueOut.idCnt, "idCnt")) {
            return H_ADDRESS_NULL;
        }
        if (uniqueConf.usePadding) {
            if (CheckInputNull(uniqueOut.idCntFill, "idCntFill")) {
                return H_ADDRESS_NULL;
            }
        }
    }

    if (CheckInputZero(uniqueIn.inputIdCnt, "inputIdCnt")) {
        return H_NUM_SMALL;
    }

    if (uniqueIn.inputIdCnt > MAX_ID_COUNT) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "inputIdCnt can not larger than 2^28");
        return H_ERROR;
    }

    return H_OK;
}

int UniqueImpl::CheckInput(UniqueIn &uniqueIn, UniqueOut &uniqueOut)
{
    if (CheckInputNull(uniqueIn.inputId, "inputId") || CheckInputNull(uniqueOut.uniqueId, "uniqueId") ||
        CheckInputNull(uniqueOut.index, "index")) {
        return H_ADDRESS_NULL;
    }
    return H_OK;
}

bool UniqueImpl::CheckInputNull(void *ptr, const std::string &name) const
{
    if (ptr == nullptr) {
        std::stringstream sm;
        sm << name << "can not be nullptr";
        ExternalLogger::PrintLog(LogLevel::ERROR, sm.str());
        return true;
    }
    return false;
}

bool UniqueImpl::CheckInputZero(int64_t in, const std::string &name) const
{
    if (in <= 0) {
        std::stringstream sm;
        sm << name << "can not be zero or negative";
        ExternalLogger::PrintLog(LogLevel::ERROR, sm.str());
        return true;
    }
    return false;
}

bool UniqueImpl::IsInitialized() const
{
    if (unique == nullptr) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "please call Initialize before DoUnique");
        return false;
    }
    return true;
}
}
}
