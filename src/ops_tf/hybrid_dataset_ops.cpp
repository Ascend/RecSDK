/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: dataset ops.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */


#include <algorithm>
#include <atomic>
#include <map>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/cfg/env.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/example/example.pb.h"

#include "key_process/key_process.h"
#include "key_process/feature_admit_and_evict.h"
#include "utils/common.h"
#include "utils/safe_queue.h"
#include "utils/singleton.h"
#include "utils/time_cost.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using namespace std;
using namespace chrono;
using namespace MxRec;

using OpKernelConstructionPtr = OpKernelConstruction*;
using OpKernelContextPtr = OpKernelContext*;
using InferenceContextPtr = ::tensorflow::shape_inference::InferenceContext*;

spdlog::stopwatch staticSw {};
spdlog::stopwatch staticReadRaw {};
array<int, MAX_CHANNEL_NUM> batchIdsInfo {};

size_t GetBatchSize(OpKernelContextPtr context, const size_t dataSize, const size_t fieldNum)
{
    if (fieldNum == 0 || dataSize / fieldNum <= 0) {
        context->SetStatus(
            errors::Aborted(__FILE__, ":", __LINE__, " ", fmt::format("batchSize error. {}/{}", dataSize, fieldNum)));
        return 0;
    }
    return dataSize / fieldNum;
}

REGISTER_OP("ClearChannel").Attr("channel_id : int");

class ClearChannel : public OpKernel {
public:
    explicit ClearChannel(OpKernelConstructionPtr context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId));

        if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("ClearChannel channelId invalid. It should be in range "
                                                           "[0, MAX_CHANNEL_NUM:{})", MAX_CHANNEL_NUM)));
            return;
        }
    }

    ~ClearChannel() = default;

    void Compute(OpKernelContextPtr context) override
    {
        spdlog::info("clear channel {}, context {}", channelId, context->step_id());
        batchIdsInfo.at(channelId) = 0;
    }

private:
    int channelId {};
};

REGISTER_KERNEL_BUILDER(Name("ClearChannel").Device(DEVICE_CPU), ClearChannel);


// ##################### ReturnTimestamp #######################
REGISTER_OP("ReturnTimestamp")
    .Input("input: int64")
    .Output("output: int64")
    .SetShapeFn([](InferenceContextPtr c) {
        c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
        return Status::OK();
    });
class ReturnTimestamp : public OpKernel {
public:
    explicit ReturnTimestamp(OpKernelConstructionPtr context) : OpKernel(context)
    {}

    ~ReturnTimestamp() = default;

    void Compute(OpKernelContextPtr context) override
    {
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
        auto out = output->flat<int64>();
        out(0) = time(nullptr);
    }
};

REGISTER_KERNEL_BUILDER(Name("ReturnTimestamp").Device(DEVICE_CPU), ReturnTimestamp);

// ##################### ReadEmbKeyV2Dynamic #######################
REGISTER_OP("ReadEmbKeyV2Dynamic")
    .Input("sample: T")
    .Input("splits: int32")
    .Output("output: int32")
    .Attr("T: {int64, int32}")
    .Attr("channel_id: int")
    .Attr("emb_name: list(string)")     // for which table to lookup
    .Attr("timestamp: bool")            // use for feature evict, (unix timestamp)
    .SetShapeFn([](InferenceContextPtr c) {
        c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
        return Status::OK();
    });

class ReadEmbKeyV2Dynamic : public OpKernel {
public:
    explicit ReadEmbKeyV2Dynamic(OpKernelConstructionPtr context) : OpKernel(context)
    {
        if (!spdlog::get("console")) {
            auto logger = spdlog::stderr_color_mt("console");
            spdlog::set_default_logger(logger);
        } else {
            spdlog::set_default_logger(spdlog::get("console"));
        }
        spdlog::cfg::load_env_levels();
        spdlog::default_logger()->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::debug("ReadEmbKeyV2Dynamic init");
        OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId)); // 0 train or 1 inference
        OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embNames));
        OP_REQUIRES_OK(context, context->GetAttr("timestamp", &isTimestamp));

        // 特征准入&淘汰功能 相关校验

        // 配置了，也不能多配、配不相关的；同时支持“准入&淘汰”，则不能没有时间戳
        if (!FeatureAdmitAndEvict::m_cfgThresholds.empty() &&
            !FeatureAdmitAndEvict::IsThresholdCfgOK(FeatureAdmitAndEvict::m_cfgThresholds, embNames, isTimestamp)) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("threshold config, or timestamp error ...")));
            return;
        }

        if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("ReadEmbKeyV2Dynamic channelId invalid. It should be in "
                                                           "range [0, MAX_CHANNEL_NUM:{})", MAX_CHANNEL_NUM)));
            return;
        }
        batchIdsInfo.at(channelId) = 0;

        const char* threadNumEnv = getenv("KEY_PROCESS_THREAD_NUM");
        if (threadNumEnv != nullptr) {
            threadNum = static_cast<int>(*threadNumEnv) - static_cast<int>('0');
            if (threadNum > KEY_PROCESS_THREAD || threadNum < 0) {
                throw runtime_error(fmt::format("{} is not valid", threadNum));
            }
        } else {
            threadNum = KEY_PROCESS_THREAD;
        }

        auto keyProcess = Singleton<KeyProcess>::GetInstance();
        if (!keyProcess->isRunning) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", "KeyProcess not running."));
            return;
        }
        maxStep = keyProcess->GetMaxStep(channelId);
    }
    ~ReadEmbKeyV2Dynamic() = default;

    void Compute(OpKernelContextPtr context) override
    {
        EASY_FUNCTION();
        spdlog::debug("enter ReadEmbKeyV2Dynamic");
        spdlog::stopwatch sw;
        int batchId = batchIdsInfo.at(channelId)++;
        if (channelId == 1) {
            if (maxStep != -1 && batchId >= maxStep) {
                spdlog::warn("skip excess batch after {}/{}", batchId, maxStep);
                return;
            }
        }
        const Tensor& inputTensor = context->input(TensorIndex::TENSOR_INDEX_0);
        const auto& splits = context->input(TENSOR_INDEX_1).flat<int32>();
        int fieldNum = 0;
        for (int i = 0; i < splits.size(); ++i) {
            fieldNum += splits(i);
        }
        size_t dataSize = inputTensor.NumElements();

        time_t timestamp = -1;
        // 如果传递了时间戳，解析和校验
        if (isTimestamp && !ParseTimestampAndCheck(inputTensor, batchId, fieldNum, timestamp, dataSize)) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("timestamp[{}] error, skip excess batch after {}/{}",
                                                           timestamp, batchId, maxStep)));
            return;
        }
        // 保证所有embNames在m_embStatus中有状态记录
        SetCurrEmbNamesStatus(embNames, FeatureAdmitAndEvict::m_embStatus);

        // [batchId % KEY_PROCESS_THREAD] which thread process this batch
        // [KEY_PROCESS_THREAD * 0 or 1] train or inference
        int batchQueueId = batchId % threadNum + KEY_PROCESS_THREAD * channelId;
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
        auto out = output->flat<int32>();
        out(0) = batchId;

        TimeCost enqueueTC;
        EnqueueBatchData(std::vector<int>{batchId, batchQueueId}, timestamp, inputTensor, splits);
        TIME_PRINT(KEY_PROCESS "ReadEmbKeyV2Dynamic read batch cost(ms):{}, elapsed from last(ms):{}, "
                        "enqueueTC(ms):{}, batch[{}]:{}",
                        Format2Ms(sw), Format2Ms(staticSw), enqueueTC.ElapsedMS(), channelId, batchId);
        staticSw.reset();
    }

    void CheckEmbTables()
    {
        auto keyProcess = Singleton<KeyProcess>::GetInstance();
        for (size_t i = 0; i < embNames.size(); ++i) {
            if (!keyProcess->hasEmbName(embNames.at(i))) {
                spdlog::info("ReadEmbKeyV2Dynamic not found emb_name:{} {}", i, embNames.at(i));
                tableUsed.push_back(false);
            } else {
                tableUsed.push_back(true);
            }
        }
    }

    void EnqueueBatchData(std::vector<int> ids, time_t timestamp,
                          const Tensor& inputTensor, const TTypes<int32>::ConstFlat& splits)
    {
        if (tableUsed.empty()) {
            CheckEmbTables();
        }
        auto queue = SingletonQueue<emb_batch_t>::getInstances(ids[1]);
        size_t offset = 0;
        if (isTimestamp) {
            offset += 1; // 前面8个字节是unix时间戳
        }
        for (int i = 0; i < splits.size(); ++i) {
            if (!tableUsed.at(i)) {
                offset += splits(i);
                continue;
            }
            auto batchData = queue->GetOne(); // get dirty or empty data block
            batchData->name = embNames.at(i);
            size_t len = splits(i);
            batchData->channel = channelId;
            batchData->batchId = ids[0];
            batchData->sample.resize(len);
            if (isTimestamp) {
                batchData->timestamp = timestamp;
            }

            if (inputTensor.dtype() == tensorflow::DT_INT32 || inputTensor.dtype() == tensorflow::DT_INT32_REF) {
                auto src = (const int32_t*)inputTensor.tensor_data().data();
                copy(src + offset, src + offset + len, batchData->sample.data());
            } else {
                auto src = (const int64_t*)inputTensor.tensor_data().data();
                copy(src + offset, src + offset + len, batchData->sample.data());
            }
            offset += len;
            queue->Pushv(move(batchData));
        }
    }

    bool ParseTimestampAndCheck(const Tensor& inputTensor, int batchId, int fieldNumTmp, time_t& timestamp,
                                size_t& dataSize)
    {
        if (dataSize - fieldNumTmp != 1) { // 说明没有传时间戳
            spdlog::error("dataSize[{}], fieldNum[{}] ...", dataSize, fieldNumTmp);
            return false;
        }

        // 前面8个字节、即占一个featureId位，是unix时间戳
        auto src = (const time_t*)inputTensor.tensor_data().data();
        std::copy(src, src + 1, &timestamp);
        spdlog::info("current batchId[{}] timestamp[{}]", batchId, timestamp);
        dataSize -= 1;

        if (timestamp <= 0) {
            spdlog::error("timestamp[{}] <= 0 ", timestamp);
            return false;
        }

        return true;
    }

    void SetCurrEmbNamesStatus(const vector<string>& embeddingNames,
                               absl::flat_hash_map<std::string, SingleEmbTableStatus>& embStatus)
    {
        for (size_t i = 0; i < embeddingNames.size(); ++i) {
            auto it = embStatus.find(embeddingNames[i]);
            // 对配置了的，进行校验
            if (it == embStatus.end()) {
                // 没有配置的，则不需要“准入&淘汰”功能
                embStatus.insert(std::pair<std::string,
                    SingleEmbTableStatus>(embeddingNames[i], SingleEmbTableStatus::SETS_NONE));
            }
        }
    }

    int channelId {};
    vector<string> embNames {};
    vector<bool> tableUsed{};
    int maxStep = 0;
    bool isTimestamp { false };
    int threadNum = 0;
};

REGISTER_KERNEL_BUILDER(Name("ReadEmbKeyV2Dynamic").Device(DEVICE_CPU), ReadEmbKeyV2Dynamic);

// ##################### ReadEmbKeyV2 #######################
REGISTER_OP("ReadEmbKeyV2")
    .Input("sample: T")
    .Output("output: int32")
    .Attr("T: {int64, int32}")
    .Attr("channel_id: int")
    .Attr("splits: list(int)")
    .Attr("emb_name: list(string)")     // for which table to lookup
    .Attr("timestamp: bool")            // use for feature evict, (unix timestamp)
    .SetShapeFn([](InferenceContextPtr c) {
        c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
        return Status::OK();
    });

class ReadEmbKeyV2 : public OpKernel {
public:
    explicit ReadEmbKeyV2(OpKernelConstructionPtr context) : OpKernel(context)
    {
        auto logger = spdlog::get("console");
        if (!logger) {
            logger = spdlog::stderr_color_mt("console");
        }
        spdlog::set_default_logger(spdlog::get("console"));

        spdlog::cfg::load_env_levels();
        spdlog::default_logger()->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::debug("ReadEmbKeyV2 init");
        OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId)); // 0 train or 1 inference
        OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embNames));
        OP_REQUIRES_OK(context, context->GetAttr("splits", &splits)); // 每个表的field Number
        OP_REQUIRES_OK(context, context->GetAttr("timestamp", &isTimestamp));
        fieldNum = accumulate(splits.begin(), splits.end(), 0);

        // 特征准入&淘汰功能 相关校验

        // 配置了，也不能多配、配不相关的；同时支持“准入&淘汰”，则不能没有时间戳
        if (!FeatureAdmitAndEvict::m_cfgThresholds.empty() &&
            !FeatureAdmitAndEvict::IsThresholdCfgOK(FeatureAdmitAndEvict::m_cfgThresholds, embNames, isTimestamp)) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("threshold config, or timestamp error ...")));
            return;
        }

        if (splits.size() != embNames.size()) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("splits & embNames size error.{} {}", splits.size(),
                                                           embNames.size())));
            return;
        }
        if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("ReadEmbKeyV2 channelId invalid. It should be in range "
                                                           "[0, MAX_CHANNEL_NUM:{})", MAX_CHANNEL_NUM)));
            return;
        }
        batchIdsInfo.at(channelId) = 0;

        const char* threadNumEnv = getenv("KEY_PROCESS_THREAD_NUM");
        if (threadNumEnv != nullptr) {
            threadNum = static_cast<int>(*threadNumEnv) - static_cast<int>('0');
            if (threadNum > KEY_PROCESS_THREAD || threadNum < 0) {
                throw runtime_error(fmt::format("{} is not valid", threadNum));
            }
        }
        auto keyProcess = Singleton<KeyProcess>::GetInstance();
        if (!keyProcess->isRunning) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", "KeyProcess not running."));
            return;
        }
        maxStep = keyProcess->GetMaxStep(channelId);
    }

    ~ReadEmbKeyV2() = default;

    void Compute(OpKernelContextPtr context) override
    {
        EASY_FUNCTION();
        spdlog::debug("enter ReadEmbKeyV2");
        spdlog::stopwatch sw;
        int batchId = batchIdsInfo.at(channelId)++;
        Tensor* output = nullptr;
        if (channelId == 1) {
            if (maxStep != -1 && batchId >= maxStep) {
                spdlog::warn("skip excess batch after {}/{}", batchId, maxStep);
                OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
                auto out = output->flat<int32>();
                out(0) = batchId;
                return;
            }
        }
        const Tensor& inputTensor = context->input(TensorIndex::TENSOR_INDEX_0);
        size_t dataSize = inputTensor.NumElements();

        time_t timestamp = -1;
        // 如果传递了时间戳，解析和校验
        if (isTimestamp && !ParseTimestampAndCheck(inputTensor, batchId, fieldNum, timestamp, dataSize)) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                               fmt::format("timestamp[{}] error, skip excess batch after {}/{}",
                                                           timestamp, batchId, maxStep)));
            return;
        }
        // 保证所有embNames在m_embStatus中有状态记录
        SetCurrEmbNamesStatus(embNames, FeatureAdmitAndEvict::m_embStatus);

        // [batchId % KEY_PROCESS_THREAD] which thread process this batch
        // [KEY_PROCESS_THREAD * 0 or 1] train or inference
        int batchQueueId = batchId % threadNum + KEY_PROCESS_THREAD * channelId;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
        auto out = output->flat<int32>();
        out(0) = batchId;

        TimeCost enqueueTC;
        EnqueueBatchData(batchId, batchQueueId, timestamp, inputTensor);
        TIME_PRINT(KEY_PROCESS "ReadEmbKeyV2Static read batch cost(ms):{}, elapsed from last(ms):{}, "
                        "enqueueTC(ms):{}, batch[{}]:{}",
                        Format2Ms(sw), Format2Ms(staticSw), enqueueTC.ElapsedMS(), channelId, batchId);
        staticSw.reset();
    }

    void CheckEmbTables()
    {
        auto keyProcess = Singleton<KeyProcess>::GetInstance();
        for (size_t i = 0; i < splits.size(); ++i) {
            if (!keyProcess->hasEmbName(embNames.at(i))) {
                spdlog::info("ReadEmbKeyV2 not found emb_name:{} {}", i, embNames.at(i));
                tableUsed.push_back(false);
            } else {
                tableUsed.push_back(true);
            }
        }
    }

    void EnqueueBatchData(int batchId, int batchQueueId, time_t timestamp, const Tensor& inputTensor)
    {
        if (tableUsed.empty()) {
            CheckEmbTables();
        }
        auto queue = SingletonQueue<emb_batch_t>::getInstances(batchQueueId);

        size_t offset = 0;
        if (isTimestamp) {
            offset += 1; // 前面8个字节是unix时间戳
        }
        for (size_t i = 0; i < splits.size(); ++i) {
            if (!tableUsed.at(i)) {
                offset += splits.at(i);
                continue;
            }
            auto batchData = queue->GetOne(); // get dirty or empty data block
            batchData->name = embNames.at(i);
            size_t len = splits.at(i);
            batchData->channel = channelId;
            batchData->batchId = batchId;
            batchData->sample.resize(len);
            if (isTimestamp) {
                batchData->timestamp = timestamp;
            }

            if (inputTensor.dtype() == tensorflow::DT_INT32 || inputTensor.dtype() == tensorflow::DT_INT32_REF) {
                auto src = (const int32_t*)inputTensor.tensor_data().data();
                copy(src + offset, src + offset + len, batchData->sample.data());
            } else {
                auto src = (const int64_t*)inputTensor.tensor_data().data();
                copy(src + offset, src + offset + len, batchData->sample.data());
            }
            offset += len;
            queue->Pushv(move(batchData));
        }
    }

    bool ParseTimestampAndCheck(const Tensor& inputTensor, int batchId, int fieldNumTmp, time_t& timestamp,
                                size_t& dataSize)
    {
        if (dataSize - fieldNumTmp != 1) { // 说明没有传时间戳
            spdlog::error("dataSize[{}], fieldNum[{}] ...", dataSize, fieldNumTmp);
            return false;
        }

        // 前面8个字节、即占一个featureId位，是unix时间戳
        auto src = (const time_t*)inputTensor.tensor_data().data();
        std::copy(src, src + 1, &timestamp);
        spdlog::info("current batchId[{}] timestamp[{}]", batchId, timestamp);
        dataSize -= 1;

        if (timestamp <= 0) {
            spdlog::error("timestamp[{}] <= 0 ", timestamp);
            return false;
        }

        return true;
    }
    void SetCurrEmbNamesStatus(const vector<string>& embeddingNames,
                               absl::flat_hash_map<std::string, SingleEmbTableStatus>& embStatus)
    {
        for (size_t i = 0; i < embeddingNames.size(); ++i) {
            auto it = embStatus.find(embeddingNames[i]);
            // 对配置了的，进行校验
            if (it == embStatus.end()) {
                // 没有配置的，则不需要“准入&淘汰”功能
                embStatus.insert(std::pair<std::string,
                    SingleEmbTableStatus>(embeddingNames[i], SingleEmbTableStatus::SETS_NONE));
            }
        }
    }

    int channelId {};
    vector<int> splits {};
    vector<bool> tableUsed{};
    int fieldNum {};
    vector<string> embNames {};
    int maxStep = 0;
    bool isTimestamp { false };
    int threadNum = KEY_PROCESS_THREAD;
};

REGISTER_KERNEL_BUILDER(Name("ReadEmbKeyV2").Device(DEVICE_CPU), ReadEmbKeyV2);

// ##################### ReadEmbKeyDatasetDummy #######################
REGISTER_OP("ReadEmbKeyDatasetDummy")
    .Input("sample: T")
    .Output("lookup_vec: int32")
    .Output("restore_vec: int32")
    .Attr("T: {int64}")
    .Attr("max_lookup_len: int")
    .SetShapeFn([](InferenceContextPtr c) {
        int temp;
        TF_RETURN_IF_ERROR(c->GetAttr("max_lookup_len", &temp));
        c->set_output(TensorIndex::TENSOR_INDEX_0, c->Vector(temp));
        c->set_output(TensorIndex::TENSOR_INDEX_1, c->input(TensorIndex::TENSOR_INDEX_0));
        return Status::OK();
    });

class ReadEmbKeyDatasetDummy : public OpKernel {
public:
    explicit ReadEmbKeyDatasetDummy(OpKernelConstructionPtr context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("max_lookup_len", &lookupLen));
    }

    ~ReadEmbKeyDatasetDummy() override = default;

    void Compute(OpKernelContextPtr context) override
    {
        EASY_FUNCTION();
        spdlog::stopwatch sw;
        const Tensor& inputTensor = context->input(TensorIndex::TENSOR_INDEX_0);
        auto input = inputTensor.flat<int64>();
        const int restoreLen = static_cast<int>(input.size());

        // write lookup & restore vec
        Tensor* lookupVec = nullptr;
        Tensor* restoreVecTensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape { lookupLen }, &lookupVec));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape { restoreLen }, &restoreVecTensor));
        auto l = lookupVec->flat<int32>();
        auto r = restoreVecTensor->flat<int32>();

        // check whether lookupLen is zero
        if (lookupLen == 0) {
            throw runtime_error("lookupLen is 0, it causes the denominator to be 0 during division");
        }

        // dummy data
        for (int i { 0 }; i < lookupLen; ++i) {
            l(i) = i;
        }
        for (int i { 0 }; i < restoreLen; ++i) {
            r(i) = i % lookupLen;
        }
        spdlog::warn("dummy read batch cost: {},elapsed from last {}",
                     Format2Ms(sw), Format2Ms(staticSw));
        staticSw.reset();
    }

    int lookupLen {};
};

REGISTER_KERNEL_BUILDER(Name("ReadEmbKeyDatasetDummy").Device(DEVICE_CPU), ReadEmbKeyDatasetDummy);

class CustOps : public OpKernel {
public:
    explicit CustOps(OpKernelConstructionPtr context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContextPtr context) override
    {
        spdlog::info("context {}", context->step_id());
        std::cout << " Cust opp not installed!!" << std::endl;
    }

    ~CustOps() = default;
};

REGISTER_OP("EmbeddingLookupByAddress")
        .Input("address: int64")
        .Attr("embedding_dim: int")
        .Attr("embedding_type: int")
        .Output("y: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle addrShape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &addrShape));
            int embSize;
            TF_RETURN_IF_ERROR(c->GetAttr("embedding_dim", &embSize));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(addrShape, 0);
            c->set_output(TENSOR_INDEX_0, c->Matrix(rows, embSize));
            return Status::OK();
        });

REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupByAddress").Device(DEVICE_CPU), CustOps);

REGISTER_OP("EmbeddingUpdateByAddress")
        .Input("address: int64")
        .Input("embedding: float")
        .Attr("update_type: int")
        .Output("y: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle addrShape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &addrShape));
            ShapeHandle embeddingShape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &embeddingShape));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(addrShape, 0);
            tensorflow::shape_inference::DimensionHandle cols = c->Dim(embeddingShape, 1);
            c->set_output(TENSOR_INDEX_0, c->Matrix(rows, cols));
            return Status::OK();
        });

REGISTER_KERNEL_BUILDER(Name("EmbeddingUpdateByAddress").Device(DEVICE_CPU), CustOps);
