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
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/cfg/env.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/example/example.pb.h"

#include "securec.h"

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
array<atomic<int>, MAX_CHANNEL_NUM> batchIdsInfo {};

REGISTER_OP("ClearChannel").Attr("channel_id : int");

class ClearChannel : public OpKernel {
public:
    explicit ClearChannel(OpKernelConstructionPtr context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId));

        if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                fmt::format("ClearChannel channelId invalid. It should be in range [0, MAX_CHANNEL_NUM:{})",
                MAX_CHANNEL_NUM)));
            return;
        }
    }

    ~ClearChannel() = default;

    void Compute(OpKernelContextPtr context) override
    {
        spdlog::info("clear channel {}", channelId);
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
    .Attr("channel_name: list(string)") // use for multi lookup
    .Attr("modify_graph: bool")         // auto modify graph enabled
    .SetShapeFn([](InferenceContextPtr c) {
        c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
        return Status::OK();
    });

class ReadEmbKeyV2Dynamic : public OpKernel {
public:
    explicit ReadEmbKeyV2Dynamic(OpKernelConstructionPtr context) : OpKernel(context)
    {
        spdlog::cfg::load_env_levels();
        spdlog::default_logger()->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::debug("ReadEmbKeyV2Dynamic init");
        OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId)); // 0 train or 1 inference
        OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embNames));
        OP_REQUIRES_OK(context, context->GetAttr("timestamp", &isTimestamp));
        OP_REQUIRES_OK(context, context->GetAttr("channel_name", &channelNames));
        OP_REQUIRES_OK(context, context->GetAttr("modify_graph", &modifyGraph));

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
                fmt::format("ReadEmbKeyV2Dynamic channelId invalid. It should be in range [0, MAX_CHANNEL_NUM:{})",
                            MAX_CHANNEL_NUM)));
            return;
        }
        batchIdsInfo.at(channelId) = 0;

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
        int batchId = batchIdsInfo.at(channelId).fetch_add(1);
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
                fmt::format("timestamp[{}] error, skip excess batch after {}/{}", timestamp, batchId, maxStep)));
            return;
        }
        // 保证所有embNames在m_embStatus中有状态记录
        SetCurrEmbNamesStatus(embNames, FeatureAdmitAndEvict::m_embStatus);

        // [batchId % PerfConfig::keyProcessThreadNum] which thread process this batch
        // [PerfConfig::keyProcessThreadNum * 0 or 1] train or inference
        int batchQueueId = batchId % PerfConfig::keyProcessThreadNum + PerfConfig::keyProcessThreadNum * channelId;
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
        auto out = output->flat<int32>();
        out(0) = batchId;
        EnqueueBatchData(std::vector<int>{batchId, batchQueueId}, timestamp, inputTensor, splits);
        TIME_PRINT(KEY_PROCESS "read batch cost: {}, elapsed from last:{}, batch[{}]:{}, "
                   "splits: {}, dataSize: {}, filedNum: {}, channelNames: {}, modifyGraph: {}",
                duration_cast<milliseconds>((sw).elapsed()), duration_cast<milliseconds>((staticSw).elapsed()),
                   channelId, batchId, splits.size(), dataSize, fieldNum, channelNames, modifyGraph);
        staticSw.reset();
    }

    void EnqueueBatchData(std::vector<int> ids, time_t timestamp,
                          const Tensor& inputTensor, const TTypes<int32>::ConstFlat& splits)
    {
        auto queue = SingletonQueue<emb_batch_t>::getInstances(ids[1]);
        size_t offset = 0;
        if (isTimestamp) {
            offset += 1; // 前面8个字节是unix时间戳
        }
        for (int i = 0; i < splits.size(); ++i) {
            auto batchData = queue->WaitAndGetOne(); // get dirty or empty data block
            batchData->name = embNames.at(i);
            if (modifyGraph) {
                batchData->modifyGraph = modifyGraph;
                batchData->channelName = channelNames.at(i);
            }
            size_t len = splits(i);
            batchData->channel = channelId;
            batchData->batchId = ids[0];
            batchData->batchSize = len;
            if (isTimestamp) {
                batchData->timestamp = timestamp;
            }
            spdlog::debug("batch[{}/{}] flatten bs: {}", ids[0], i+1, len);
            std::unique_ptr<emb_batch_t> batch = TensorCopy(inputTensor, move(batchData), len, offset);
            if (batch == nullptr) {
                spdlog::error("batch can not be null");
                return;
            }
            queue->Pushv(move(batch));
        }
        TIME_PRINT(KEY_PROCESS "EnqueueBatchData, batchId:{}, channelId:{}", ids[0], channelId);
    }

    std::unique_ptr<emb_batch_t> TensorCopy(const Tensor& inputTensor, std::unique_ptr<emb_batch_t> batchData,
                                            const size_t& len, size_t& offset)
    {
        if (len == 0) {
            spdlog::error("len can not be zero");
            return nullptr;
        }
        TimeCost ct;
        void* src = nullptr;
        size_t memSize;
        if (inputTensor.dtype() == tensorflow::DT_INT32 || inputTensor.dtype() == tensorflow::DT_INT32_REF) {
            batchData->isInt64 = false;
            memSize = len * sizeof(int32_t);
            src = reinterpret_cast<void *>(
                    reinterpret_cast<int32_t *>(const_cast<string *>((string *)(inputTensor.tensor_data().data()))) +
                    offset);
        } else {
            batchData->isInt64 = true;
            memSize = len * sizeof(int64_t);
            src = reinterpret_cast<void *>(
                    reinterpret_cast<int64_t *>(const_cast<string *>((string *)(inputTensor.tensor_data().
                            data()))) + offset);
        }
        batchData->tensorAddr = malloc(memSize);
        if (batchData->tensorAddr == nullptr) {
            spdlog::error("mmemory allocation failded...");
        }
        void* dst = reinterpret_cast<void *>(batchData->tensorAddr);
        auto rc = memcpy_s(dst, memSize, src, memSize);
        if (rc != 0) {
            spdlog::error("[ReadEmbKeyV2Dynamic]memcpy_s failded... memSize: {}", memSize);
        }
        TIME_PRINT("copy TimeCost(ms):{}", ct.ElapsedMS());
        offset += len;
        return move(batchData);
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
    vector<string> channelNames {};
    int maxStep = 0;
    bool isTimestamp { false };
    bool modifyGraph { false };
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
    .Attr("channel_name: list(string)") // use for multi lookup
    .Attr("modify_graph: bool")         // auto modify graph enabled
    .SetShapeFn([](InferenceContextPtr c) {
        c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
        return Status::OK();
    });

class ReadEmbKeyV2 : public OpKernel {
public:
    explicit ReadEmbKeyV2(OpKernelConstructionPtr context) : OpKernel(context)
    {
        spdlog::cfg::load_env_levels();
        spdlog::default_logger()->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        spdlog::debug("ReadEmbKeyV2 init");
        OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId)); // 0 train or 1 inference
        OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embNames));
        OP_REQUIRES_OK(context, context->GetAttr("splits", &splits)); // 每个表的field Number
        OP_REQUIRES_OK(context, context->GetAttr("timestamp", &isTimestamp));
        OP_REQUIRES_OK(context, context->GetAttr("channel_name", &channelNames));
        OP_REQUIRES_OK(context, context->GetAttr("modify_graph", &modifyGraph));
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
                fmt::format("splits & embNames size error.{} {}", splits.size(), embNames.size())));
            return;
        }
        if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
            context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                fmt::format("ReadEmbKeyV2 channelId invalid. It should be in range [0, MAX_CHANNEL_NUM:{})",
                            MAX_CHANNEL_NUM)));
            return;
        }
        batchIdsInfo.at(channelId) = 0;

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
                fmt::format("timestamp[{}] error, skip excess batch after {}/{}", timestamp, batchId, maxStep)));
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
            auto out = output->flat<int32>();
            out(0) = batchId;
            return;
        }
        // 保证所有embNames在m_embStatus中有状态记录
        SetCurrEmbNamesStatus(embNames, FeatureAdmitAndEvict::m_embStatus);

        // [batchId % PerfConfig::keyProcessThreadNum] which thread process this batch
        // [PerfConfig::keyProcessThreadNum * 0 or 1] train or inference
        int batchQueueId = batchId % PerfConfig::keyProcessThreadNum + PerfConfig::keyProcessThreadNum * channelId;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
        auto out = output->flat<int32>();
        out(0) = batchId;

        TimeCost tc;
        EnqueueBatchData(batchId, batchQueueId, timestamp, inputTensor);
        TIME_PRINT("EnqueueBatchData TimeCost(ms):{}", tc.ElapsedMS());

        TIME_PRINT(KEY_PROCESS
        "read batch cost: {}, elapsed from last:{}, batch[{}]:{}", duration_cast<milliseconds>((sw).elapsed()),
                duration_cast<milliseconds>((staticSw).elapsed()), channelId, batchId);
        staticSw.reset();
    }

    int EnqueueBatchData(int batchId, int batchQueueId, time_t timestamp, const Tensor& inputTensor)
    {
        auto queue = SingletonQueue<emb_batch_t>::getInstances(batchQueueId);
        size_t offset = 0;
        if (isTimestamp) {
            offset += 1; // 前面8个字节是unix时间戳
        }
        TimeCost ctAll;
        for (size_t i = 0; i < splits.size(); ++i) {
            TimeCost tp;
            auto batchData = queue->WaitAndGetOne(); // get dirty or empty data block
            TIME_PRINT("TryPopTimeCost(ms):{}", tp.ElapsedMS());

            batchData->name = embNames.at(i);
            if (modifyGraph) {
                batchData->modifyGraph = modifyGraph;
                batchData->channelName = channelNames.at(i);
            }
            size_t len = splits.at(i);
            batchData->channel = channelId;
            batchData->batchId = batchId;
            batchData->batchSize = len;
            TimeCost fz;
            if (isTimestamp) {
                batchData->timestamp = timestamp;
            }
            TIME_PRINT("fz TimeCost(ms):{}", fz.ElapsedMS());

            std::unique_ptr<emb_batch_t> batch = TensorCopy(inputTensor, move(batchData), len, offset);
            if (batch == nullptr) {
                spdlog::error("batch can not be null");
                return -1;
            }
            queue->Pushv(move(batch));
        }
        TIME_PRINT("all copy TimeCost(ms):{}", ctAll.ElapsedMS());
        return 0;
    }

    std::unique_ptr<emb_batch_t> TensorCopy(const Tensor& inputTensor, std::unique_ptr<emb_batch_t> batchData,
                                            const size_t& len, size_t& offset)
    {
        if (len == 0) {
            spdlog::error("len can not be zero");
            return nullptr;
        }
        TimeCost ct;
        void* src = nullptr;
        size_t memSize;
        if (inputTensor.dtype() == tensorflow::DT_INT32 || inputTensor.dtype() == tensorflow::DT_INT32_REF) {
            batchData->isInt64 = false;
            memSize = len * sizeof(int32_t);
            src = reinterpret_cast<void *>(
                    reinterpret_cast<int32_t *>(const_cast<string *>((string *)(inputTensor.tensor_data().data()))) +
                    offset);
        } else {
            batchData->isInt64 = true;
            memSize = len * sizeof(int64_t);
            src = reinterpret_cast<void *>(
                    reinterpret_cast<int64_t *>(const_cast<string *>((string *)(inputTensor.tensor_data().data()))) +
                    offset);
        }
        batchData->tensorAddr = malloc(memSize);
        if (batchData->tensorAddr == nullptr) {
            spdlog::error("mmemory allocation failded...");
        }
        void* dst = reinterpret_cast<void *>(batchData->tensorAddr);
        auto rc = memcpy_s(dst, memSize, src, memSize);
        if (rc != 0) {
            spdlog::error("[ReadEmbKeyV2Static]memcpy_s failded... memSize: {}", memSize);
        }
        TIME_PRINT("copy TimeCost(ms):{}", ct.ElapsedMS());
        offset += len;
        return move(batchData);
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
    int fieldNum {};
    vector<string> embNames {};
    vector<string> channelNames {};
    int maxStep = 0;
    bool isTimestamp { false };
    bool modifyGraph { false };
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

        // dummy data
        for (int i { 0 }; i < lookupLen; ++i) {
            l(i) = i;
        }
        for (int i { 0 }; i < restoreLen; ++i) {
            r(i) = i % lookupLen;
        }
        spdlog::warn("dummy read batch cost: {},elapsed from last {}", duration_cast<milliseconds>((sw).elapsed()),
                     duration_cast<milliseconds>((staticSw).elapsed()));
        staticSw.reset();
    }

    int lookupLen {};
};

REGISTER_KERNEL_BUILDER(Name("ReadEmbKeyDatasetDummy").Device(DEVICE_CPU), ReadEmbKeyDatasetDummy);


// ##################### ReadRaw #######################
REGISTER_OP("ReadRaw")
    .Input("sample: string")
    .Output("int_output: int64")
    .Output("float_output: float")
    .Attr("int_len: int")
    .Attr("float_len: int")
    .Attr("feat_order: list(string)")
    .SetShapeFn([](InferenceContextPtr c) {
        int temp;
        TF_RETURN_IF_ERROR(c->GetAttr("int_len", &temp));
        c->set_output(TENSOR_INDEX_0, c->Vector(temp));
        TF_RETURN_IF_ERROR(c->GetAttr("float_len", &temp));
        c->set_output(TENSOR_INDEX_1, c->Vector(temp));
        return Status::OK();
    });

class ReadRaw : public OpKernel {
public:
    explicit ReadRaw(OpKernelConstructionPtr context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("int_len", &intLen));
        OP_REQUIRES_OK(context, context->GetAttr("float_len", &floatLen));
        OP_REQUIRES_OK(context, context->GetAttr("feat_order", &featOrder));
        sampleId = 0;
    }

    ~ReadRaw() override = default;

    void Compute(OpKernelContextPtr context) override
    {
        spdlog::stopwatch sw;
        Tensor* intTensor = nullptr;
        Tensor* floatTensor = nullptr;
        int intDataIndex = 0;
        int floatDataIndex = 0;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape { intLen }, &intTensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape { floatLen }, &floatTensor));
        const Tensor& inputTensor = context->input(TENSOR_INDEX_0);
        auto input = inputTensor.flat<tstring>()(0);
        tensorflow::Example example;
        if (!example.ParseFromString(input)) {
            cerr << "Failed to parse file." << endl;
        }
        spdlog::stopwatch sw_copy;
        auto all_feature_map = example.features().feature();
        for (const auto& featName: featOrder) {
            auto& cur_feature_value = all_feature_map.at(featName);
            if (cur_feature_value.has_int64_list()) {
                auto int64List = cur_feature_value.int64_list();
                int64* flat = intTensor->flat<int64>().data() + intDataIndex;
                std::copy(int64List.value().begin(), int64List.value().end(), flat);
                intDataIndex += int64List.value_size();
            }
            if (cur_feature_value.has_float_list()) {
                auto floatList = cur_feature_value.float_list();
                float* flat = floatTensor->flat<float>().data() + floatDataIndex;
                std::copy(floatList.value().begin(), floatList.value().end(), flat);
                floatDataIndex += floatList.value_size();
            }
        }
        spdlog::info("ReadRaw sampleId:{} cost:{} copy:{} , elapsed from last:{}", sampleId++,
                     duration_cast<milliseconds>((sw).elapsed()),
                     duration_cast<milliseconds>((sw_copy).elapsed()),
                     duration_cast<milliseconds>((staticReadRaw).elapsed()));
        staticReadRaw.reset();
    }

    int intLen;
    int floatLen;
    vector<string> featOrder;
    atomic<int> sampleId;
};

REGISTER_KERNEL_BUILDER(Name("ReadRaw").Device(DEVICE_CPU), ReadRaw);


// ##################### ReadRawDummy #######################
REGISTER_OP("ReadRawDummy")
    .Input("sample: int64")
    .Output("int_output: int64")
    .Output("float_output: float")
    .Attr("int_len: int")
    .Attr("float_len: int")
    .SetShapeFn([](InferenceContextPtr c) {
        int temp;
        TF_RETURN_IF_ERROR(c->GetAttr("int_len", &temp));
        c->set_output(TENSOR_INDEX_0, c->Vector(temp));
        TF_RETURN_IF_ERROR(c->GetAttr("float_len", &temp));
        c->set_output(TENSOR_INDEX_1, c->Vector(temp));
        return Status::OK();
    });

class ReadRawDummy : public OpKernel {
public:
    explicit ReadRawDummy(OpKernelConstructionPtr context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("int_len", &intLen));
        OP_REQUIRES_OK(context, context->GetAttr("float_len", &floatLen));
    }

    ~ReadRawDummy() override = default;

    void Compute(OpKernelContextPtr context) override
    {
        spdlog::stopwatch sw;
        Tensor* intTensor = nullptr;
        Tensor* floatTensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape { intLen }, &intTensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape { floatLen }, &floatTensor));

        const Tensor& inputTensor = context->input(TENSOR_INDEX_0);
        auto input = inputTensor.flat<int64>();
        int32_t batchId = input(0);

        spdlog::info("ReadRawDummy cost:{}, elapsed from last:{} , batchId = {}",
                     duration_cast<milliseconds>((sw).elapsed()),
                     duration_cast<milliseconds>((staticReadRaw).elapsed()), batchId);
        staticReadRaw.reset();
    }

    int intLen;
    int floatLen;
};

REGISTER_KERNEL_BUILDER(Name("ReadRawDummy").Device(DEVICE_CPU), ReadRawDummy);

class CustOps : public OpKernel {
public:
    explicit CustOps(OpKernelConstructionPtr context) : OpKernel(context)
    {
    }

    void Compute(OpKernelContextPtr context) override
    {
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