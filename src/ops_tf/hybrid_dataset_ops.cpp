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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

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

namespace {
    static TimeCost g_staticSw{};
}

namespace MxRec {
    class ClearChannel : public OpKernel {
    public:
        explicit ClearChannel(OpKernelConstructionPtr context) : OpKernel(context)
        {
            LOG_INFO("clear channel init");
            OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId));

            if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", StringFormat(
                    "ClearChannel channelId invalid. It should be in range [0, MAX_CHANNEL_NUM:%d)",
                    MAX_CHANNEL_NUM)));
                return;
            }
        }

        ~ClearChannel() override = default;

        void Compute(OpKernelContextPtr context) override
        {
            LOG_INFO("clear channel {}, context {}", channelId, context->step_id());
            HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
            hybridMgmtBlock->ResetAll(channelId);
        }

    private:
        int channelId {};
    };

    class SetThreshold : public OpKernel {
    public:
        explicit SetThreshold(OpKernelConstruction* context) : OpKernel(context)
        {
            LOG_INFO("SetThreshold init");
            OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embName));
            OP_REQUIRES_OK(context, context->GetAttr("ids_name", &idsName)); // sparse_lookup查询
        }

        ~SetThreshold() override = default;

        void Compute(OpKernelContextPtr context) override
        {
            LOG_DEBUG("enter SetThreshold");
            int threshold = 1;
            const Tensor& inputTensor = context->input(TensorIndex::TENSOR_INDEX_0);

            int available = ParseThresholdAndCheck(inputTensor, threshold);
            if (available == 0) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ",
                                                   StringFormat("threshold[%d] error", threshold)));
                return;
            }

            // 开了准入才能调用修改阈值算子
            if (!FeatureAdmitAndEvict::m_cfgThresholds.empty()) {
                auto keyProcess = Singleton<KeyProcess>::GetInstance();
                if (!keyProcess->isRunning) {
                    context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", "KeyProcess not running."));
                    return;
                }

                if (!keyProcess->GetFeatAdmitAndEvict().SetTableThresholds(threshold, embName)) {
                    context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", "threshold set error ...")
                    );
                    return;
                }
            } else {
                LOG_WARN("SetThreshold failed, because feature admit-and-evict switch is closed");
            }

            Tensor* output = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
            auto out = output->flat<int32>();
            out(0) = available;
        }

        int ParseThresholdAndCheck(const Tensor& inputTensor, int& threshold) const
        {
            // 前面8个字节、即占一个featureId位，是unix时间戳
            auto src = reinterpret_cast<const int*>(inputTensor.tensor_data().data());
            std::copy(src, src + 1, &threshold);

            if (threshold < 0) {
                LOG_ERROR("set threshold[{}] < 0 ", threshold);
                return 0;
            }
            LOG_INFO("ParseThresholdAndCheck, emb_name:[{}], ids_name: [{}], threshold: [{}]",
                     embName, idsName, threshold);

            return 1;
        }

    private:
        string embName {};
        string idsName {};
    };

    class ReturnTimestamp : public OpKernel {
    public:
        explicit ReturnTimestamp(OpKernelConstructionPtr context) : OpKernel(context)
        {}

        ~ReturnTimestamp() override = default;

        void Compute(OpKernelContextPtr context) override
        {
            Tensor* output = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
            auto out = output->flat<int64>();
            out(0) = time(nullptr);
        }
    };

    class ReadEmbKeyV2Dynamic : public OpKernel {
    public:
        explicit ReadEmbKeyV2Dynamic(OpKernelConstructionPtr context) : OpKernel(context)
        {
            LOG_DEBUG("ReadEmbKeyV2Dynamic init");
            OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId)); // 0 train or 1 inference
            OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embNames));
            OP_REQUIRES_OK(context, context->GetAttr("timestamp", &isTimestamp));
            hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
            // 特征准入&淘汰功能 相关校验

            // 配置了，也不能多配、配不相关的；同时支持“准入&淘汰”，则不能没有时间戳
            if (!FeatureAdmitAndEvict::m_cfgThresholds.empty() &&
                !FeatureAdmitAndEvict::IsThresholdCfgOK(
                    FeatureAdmitAndEvict::m_cfgThresholds, embNames, isTimestamp)
                    ) {
                context->SetStatus(
                    errors::Aborted(__FILE__, ":", __LINE__, " ", "threshold config, or timestamp error ..."));
                return;
            }

            if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", StringFormat(
                    "ReadEmbKeyV2Dynamic channelId invalid. It should be in range [0, MAX_CHANNEL_NUM:%d)",
                    MAX_CHANNEL_NUM)));
                return;
            }
            LOG_INFO(HYBRID_BLOCKING + " reset channel {}", channelId);
            hybridMgmtBlock->ResetAll(channelId);

            threadNum = GetThreadNumEnv();
            auto keyProcess = Singleton<KeyProcess>::GetInstance();
            if (!keyProcess->isRunning) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", "KeyProcess not running."));
                return;
            }
            maxStep = keyProcess->GetMaxStep(channelId);
        }
        ~ReadEmbKeyV2Dynamic() override = default;

        void Compute(OpKernelContextPtr context) override
        {
            EASY_FUNCTION();
            LOG_DEBUG("enter ReadEmbKeyV2Dynamic");
            TimeCost tc = TimeCost();
            int batchId = hybridMgmtBlock->readEmbedBatchId[channelId]++;
            Tensor* output = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
            auto out = output->flat<int32>();
            out(0) = batchId;
            if (channelId == 1) {
                if (maxStep != -1 && batchId >= maxStep) {
                    LOG_WARN("skip excess batch after {}/{}", batchId, maxStep);
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
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", StringFormat(
                    "timestamp[%d] error, skip excess batch after %d/%d", timestamp, batchId, maxStep)));
                return;
            }
            // 保证所有embNames在m_embStatus中有状态记录
            SetCurrEmbNamesStatus(embNames, FeatureAdmitAndEvict::m_embStatus);

            // [batchId % KEY_PROCESS_THREAD] which thread process this batch
            // [KEY_PROCESS_THREAD * 0 or 1] train or inference
            int batchQueueId = (batchId % threadNum) + (MAX_KEY_PROCESS_THREAD * channelId);

            TimeCost enqueueTC;
            EnqueueBatchData(std::vector<int>{batchId, batchQueueId}, timestamp, inputTensor, splits);
            LOG_DEBUG(KEY_PROCESS "ReadEmbKeyV2Dynamic read batch cost(ms):{}, elapsed from last(ms):{},"
                                  " enqueueTC(ms):{}, batch[{}]:{}",
                    tc.ElapsedMS(), g_staticSw.ElapsedMS(), enqueueTC.ElapsedMS(), channelId, batchId);
            g_staticSw = TimeCost();
        }

        void CheckEmbTables()
        {
            auto keyProcess = Singleton<KeyProcess>::GetInstance();
            for (size_t i = 0; i < embNames.size(); ++i) {
                if (!keyProcess->HasEmbName(embNames.at(i))) {
                    LOG_INFO("ReadEmbKeyV2Dynamic not found emb_name:{} {}", i, embNames.at(i));
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
            auto queue = SingletonQueue<EmbBatchT>::GetInstances(ids[1]);
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
                    auto src = reinterpret_cast<const int32_t*>(inputTensor.tensor_data().data());
                    copy(src + offset, src + offset + len, batchData->sample.data());
                } else {
                    auto src =  reinterpret_cast<const int64_t*>(inputTensor.tensor_data().data());
                    copy(src + offset, src + offset + len, batchData->sample.data());
                }
                offset += len;
                queue->Pushv(move(batchData));
            }
        }

        bool ParseTimestampAndCheck(const Tensor& inputTensor, int batchId, int fieldNumTmp, time_t& timestamp,
                                    size_t& dataSize) const
        {
            if (dataSize - fieldNumTmp != 1) { // 说明没有传时间戳
                LOG_ERROR("dataSize[{}], fieldNum[{}] ...", dataSize, fieldNumTmp);
                return false;
            }

            // 前面8个字节、即占一个featureId位，是unix时间戳
            auto src = reinterpret_cast<const time_t*>(inputTensor.tensor_data().data());
            std::copy(src, src + 1, &timestamp);
            LOG_INFO("current batchId[{}] timestamp[{}]", batchId, timestamp);
            dataSize -= 1;

            if (timestamp <= 0) {
                LOG_ERROR("timestamp[{}] <= 0 ", timestamp);
                return false;
            }

            return true;
        }

        void SetCurrEmbNamesStatus(const vector<string>& embeddingNames,
                                   absl::flat_hash_map<std::string, SingleEmbTableStatus>& embStatus) const
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
        HybridMgmtBlock* hybridMgmtBlock;
    };

    class ReadEmbKeyV2 : public OpKernel {
    public:
        explicit ReadEmbKeyV2(OpKernelConstructionPtr context) : OpKernel(context)
        {
            LOG_DEBUG("ReadEmbKeyV2 init");
            OP_REQUIRES_OK(context, context->GetAttr("channel_id", &channelId)); // 0 train or 1 inference
            OP_REQUIRES_OK(context, context->GetAttr("emb_name", &embNames));
            OP_REQUIRES_OK(context, context->GetAttr("splits", &splits)); // 每个表的field Number
            OP_REQUIRES_OK(context, context->GetAttr("timestamp", &isTimestamp));
            fieldNum = accumulate(splits.begin(), splits.end(), 0);

            hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
            // 特征准入&淘汰功能 相关校验

            // 配置了，也不能多配、配不相关的；同时支持“准入&淘汰”，则不能没有时间戳
            if (!FeatureAdmitAndEvict::m_cfgThresholds.empty() &&
                !FeatureAdmitAndEvict::IsThresholdCfgOK(FeatureAdmitAndEvict::m_cfgThresholds, embNames, isTimestamp)) {
                context->SetStatus(
                    errors::Aborted(__FILE__, ":", __LINE__, " ", "threshold config, or timestamp error ...")
                );
                return;
            }

            if (splits.size() != embNames.size()) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", StringFormat(
                    "splits & embNames size error.%d %d", splits.size(), embNames.size())));
                return;
            }
            if (channelId < 0 || channelId >= MAX_CHANNEL_NUM) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", StringFormat(
                    "ReadEmbKeyV2 channelId invalid. It should be in range [0, MAX_CHANNEL_NUM:%d)",
                    MAX_CHANNEL_NUM)));
                return;
            }
            LOG_INFO(HYBRID_BLOCKING + " reset channel {}", channelId);
            // 重置此数据通道中所有的步数
            hybridMgmtBlock->ResetAll(channelId);

            threadNum = GetThreadNumEnv();
            auto keyProcess = Singleton<KeyProcess>::GetInstance();
            if (!keyProcess->isRunning) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", "KeyProcess not running."));
                return;
            }
            maxStep = keyProcess->GetMaxStep(channelId);
        }

        ~ReadEmbKeyV2() override = default;

        void Compute(OpKernelContextPtr context) override
        {
            EASY_FUNCTION();
            LOG_DEBUG("enter ReadEmbKeyV2");
            TimeCost tc = TimeCost();
            int batchId = hybridMgmtBlock->readEmbedBatchId[channelId]++;
            Tensor* output = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape {}, &output));
            auto out = output->flat<int32>();
            out(0) = batchId;
            if (channelId == 1) {
                if (maxStep != -1 && batchId >= maxStep) {
                    LOG_WARN(StringFormat("skip excess batch after {}/{}", batchId, maxStep));
                    return;
                }
            }
            const Tensor& inputTensor = context->input(TensorIndex::TENSOR_INDEX_0);
            size_t dataSize = inputTensor.NumElements();

            time_t timestamp = -1;
            // 如果传递了时间戳，解析和校验
            if (isTimestamp && !ParseTimestampAndCheck(inputTensor, batchId, fieldNum, timestamp, dataSize)) {
                context->SetStatus(errors::Aborted(__FILE__, ":", __LINE__, " ", StringFormat(
                    "timestamp[%d] error, skip excess batch after %d/%d", timestamp, batchId, maxStep)));
                return;
            }
            // 保证所有embNames在m_embStatus中有状态记录
            SetCurrEmbNamesStatus(embNames, FeatureAdmitAndEvict::m_embStatus);

            // [batchId % KEY_PROCESS_THREAD] which thread process this batch
            // [KEY_PROCESS_THREAD * 0 or 1] train or inference
            int batchQueueId = (batchId % threadNum) + (MAX_KEY_PROCESS_THREAD * channelId);

            TimeCost enqueueTC;
            EnqueueBatchData(batchId, batchQueueId, timestamp, inputTensor);
            LOG_DEBUG(KEY_PROCESS "ReadEmbKeyV2Static read batch cost(ms):{}, elapsed from last(ms):{},"
                                  " enqueueTC(ms):{}, batch[{}]:{}",
                    tc.ElapsedMS(), g_staticSw.ElapsedMS(), enqueueTC.ElapsedMS(), channelId, batchId);
            g_staticSw = TimeCost();
        }

        void CheckEmbTables()
        {
            auto keyProcess = Singleton<KeyProcess>::GetInstance();
            for (size_t i = 0; i < splits.size(); ++i) {
                if (!keyProcess->HasEmbName(embNames.at(i))) {
                    LOG_INFO("ReadEmbKeyV2 not found emb_name:{} {}", i, embNames.at(i));
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
            auto queue = SingletonQueue<EmbBatchT>::GetInstances(batchQueueId);

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
                    auto src = reinterpret_cast<const int32_t*>(inputTensor.tensor_data().data());
                    copy(src + offset, src + offset + len, batchData->sample.data());
                } else {
                    auto src = reinterpret_cast<const int64_t*>(inputTensor.tensor_data().data());
                    copy(src + offset, src + offset + len, batchData->sample.data());
                }
                offset += len;
                queue->Pushv(move(batchData));
            }
        }

        bool ParseTimestampAndCheck(const Tensor& inputTensor, int batchId, int fieldNumTmp, time_t& timestamp,
                                    size_t& dataSize) const
        {
            if (dataSize - fieldNumTmp != 1) { // 说明没有传时间戳
                LOG_ERROR("dataSize[{}], fieldNum[{}] ...", dataSize, fieldNumTmp);
                return false;
            }

            // 前面8个字节、即占一个featureId位，是unix时间戳
            auto src = reinterpret_cast<const time_t*>(inputTensor.tensor_data().data());
            std::copy(src, src + 1, &timestamp);
            LOG_INFO("current batchId[{}] timestamp[{}]", batchId, timestamp);
            dataSize -= 1;

            if (timestamp <= 0) {
                LOG_ERROR("timestamp[{}] <= 0 ", timestamp);
                return false;
            }

            return true;
        }
        void SetCurrEmbNamesStatus(const vector<string>& embeddingNames,
                                   absl::flat_hash_map<std::string, SingleEmbTableStatus>& embStatus) const
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
        HybridMgmtBlock* hybridMgmtBlock;
    };

    class CustOps : public OpKernel {
    public:
        explicit CustOps(OpKernelConstructionPtr context) : OpKernel(context)
        {
        }

        void Compute(OpKernelContextPtr context) override
        {
            LOG_INFO("context {}", context->step_id());
            std::cout << " Cust opp not installed!!" << std::endl;
        }

        ~CustOps() override = default;
    };

}
REGISTER_OP("ClearChannel").Attr("channel_id : int");
REGISTER_KERNEL_BUILDER(Name("ClearChannel").Device(DEVICE_CPU), MxRec::ClearChannel);

// ##################### SetThreshold #######################
REGISTER_OP("SetThreshold")
.Input("input: int32")
.Attr("emb_name: string = ''")
.Attr("ids_name: string = ''")
.Output("output: int32")
.SetShapeFn([](InferenceContextPtr c) {
c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
return Status::OK();
});
REGISTER_KERNEL_BUILDER(Name("SetThreshold").Device(tensorflow::DEVICE_CPU), SetThreshold);

// ##################### ReturnTimestamp #######################
REGISTER_OP("ReturnTimestamp")
.Input("input: int64")
.Output("output: int64")
.SetShapeFn([](InferenceContextPtr c) {
c->set_output(TensorIndex::TENSOR_INDEX_0, c->Scalar());
return Status::OK();
});
REGISTER_KERNEL_BUILDER(Name("ReturnTimestamp").Device(DEVICE_CPU), MxRec::ReturnTimestamp);

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

REGISTER_KERNEL_BUILDER(Name("ReadEmbKeyV2Dynamic").Device(DEVICE_CPU), MxRec::ReadEmbKeyV2Dynamic);

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

REGISTER_KERNEL_BUILDER(Name("ReadEmbKeyV2").Device(DEVICE_CPU), MxRec::ReadEmbKeyV2);

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

REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupByAddress").Device(DEVICE_CPU), MxRec::CustOps);

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

REGISTER_KERNEL_BUILDER(Name("EmbeddingUpdateByAddress").Device(DEVICE_CPU), MxRec::CustOps);