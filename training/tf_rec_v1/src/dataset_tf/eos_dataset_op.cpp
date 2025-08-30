/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "eos_dataset_op.h"

#include <cstdint>
#include <mpi.h>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

#if defined(TF_VERSION_TF2)
#include "tensorflow/core/data/name_utils.h"
#endif

#include "key_process/key_process.h"
#include "log/logger.h"
#include "utils/common.h"
#include "utils/singleton.h"
#include "utils/time_cost.h"

using namespace std;
using namespace MxRec;

namespace tensorflow {
namespace data {

MPI_Group g_group;
MPI_Comm g_comm[2];

int g_rankId;
int g_rankSize;
int g_datasetId[2] = {0, 0};

constexpr const char *const EosDatasetOp::kDatasetType;
constexpr const char *const EosDatasetOp::kInputDataset;
constexpr const char *const EosDatasetOp::kChannelId;
constexpr const char *const EosDatasetOp::kMaxTrainSteps;
constexpr const char *const EosDatasetOp::kMaxEvalSteps;
constexpr const char *const EosDatasetOp::kOutputTypes;
constexpr const char *const EosDatasetOp::kOutputShapes;

int CheckCommFinished(MPI_Request& req, int channelId)
{
    TimeCost tc;

    while (true) {
        int flag;
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        if (flag > 0) {
            return 0;
        }

        if (tc.ElapsedSec() >= EOS_TIMEOUT) {
            tc = TimeCost();
            LOG_DEBUG("Channel: {} all_reduce execute timeout, invoked by rank: {}", channelId, g_rankId);
        }
    }
}

// 表示数据集的不可变性定义，这个类的 MakeIterator() 方法告诉 TensorFlow 怎样在数据集上生成迭代器对象。
class EosDatasetOp::Dataset : public DatasetBase {
public:
    explicit Dataset(OpKernelContext *ctx, const DatasetBase *input, int32_t channelId,
                     int32_t maxTrainSteps,
                     int32_t maxEvalSteps,
                     const DataTypeVector& outputTypes,
                     const std::vector<PartialTensorShape>& outputShapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          channelId_(channelId),
          maxTrainSteps_(maxTrainSteps),
          maxEvalSteps_(maxEvalSteps),
          outputTypes_(outputTypes),
          outputShapes_(outputShapes),
          id_(g_datasetId[channelId]) {
        input_->Ref();

        MPI_Comm_group(MPI_COMM_WORLD, &g_group);
        MPI_Comm_create(MPI_COMM_WORLD, g_group, &g_comm[channelId]);
        MPI_Comm_rank(g_comm[channelId], &g_rankId);
        MPI_Comm_size(g_comm[channelId], &g_rankSize);

        LOG_DEBUG("EosDataset: {} was born for channel: {}, maxTrainSteps: {}, maxEvalSteps: {}.",
                  g_datasetId[channelId], channelId, maxTrainSteps, maxEvalSteps);
        g_datasetId[channelId] += 1;
    }

    Dataset(const Dataset &) = delete;

    Dataset &operator=(const Dataset &) = delete;

    ~Dataset() override
    {
        LOG_DEBUG("EosDataset: {} for channel: {} has been destroied!", id_, channelId_);
        input_->Unref();
    }

    std::unique_ptr <IteratorBase> MakeIteratorInternal(const string &prefix) const override
    {
#if defined(TF_VERSION_TF2)
        string prefix_para = name_utils::IteratorPrefix(kDatasetType, prefix);
#else
        string prefix_para = prefix + "::" + kDatasetType;
#endif
        return absl::make_unique<Iterator>(Iterator::Params{
                this, prefix_para});
    }

    const DataTypeVector& output_dtypes() const override
    {
        return outputTypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override
    {
        return outputShapes_;
    }

    string DebugString() const override
    {
#if defined(TF_VERSION_TF2)
        return name_utils::DatasetDebugString(kDatasetType);
#else
        return "NpuMapDatasetOp::DataSet";
#endif
    }

    int64 Cardinality() const override
    {
        return input_->Cardinality();
    }

    Status CheckExternalState() const override
    {
        return input_->CheckExternalState();
    }

protected:
    Status
    AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b,
                       Node **output) const override
                       {
        Node *input_graph = nullptr;
        TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
        Node *channel_id_x = nullptr;
        TF_RETURN_IF_ERROR(b->AddScalar(channelId_, &channel_id_x));
        Node *max_train_steps_x = nullptr;
        TF_RETURN_IF_ERROR(b->AddScalar(maxTrainSteps_, &max_train_steps_x));
        Node *max_eval_steps_x = nullptr;
        TF_RETURN_IF_ERROR(b->AddScalar(maxEvalSteps_, &max_eval_steps_x));
        TF_RETURN_IF_ERROR(
            b->AddDataset(this, {input_graph, channel_id_x, max_train_steps_x, max_eval_steps_x},
                output));
        return Status::OK();
    }

private:
    // 表示特定数据集上的迭代器的可变性，这个类的 GetNextInternal() 方法告诉 TensorFlow 怎样获取迭代器的下一个元素。
    class Iterator : public DatasetIterator<Dataset> {
    public:
        explicit Iterator(const Params &params) : DatasetIterator<Dataset>(params), i_(0),
                                                  iter_times_(0) {}

#if defined(TF_VERSION_TF2)
        Status Initialize(IteratorContext* ctx) override
        {
            return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
        }
#else

        Status Initialize(IteratorContext *ctx) override
        {
            return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
        }

#endif
        Status
        GetNextInternal(IteratorContext *ctx, std::vector <Tensor> *out_tensors,
                        bool *end_of_sequence) override
                        {
            mutex_lock l(mu_);
            if (!input_impl_) {
                *end_of_sequence = true;
                return Status::OK();
            }
            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, out_tensors, end_of_sequence));

            auto channelId = dataset()->channelId_;
            PrintOutput(out_tensors, channelId);

            auto keyProcess = Singleton<KeyProcess>::GetInstance();
            if (channelId == 0 && iter_times_ == dataset()->maxTrainSteps_) {
                *end_of_sequence = true;
            }
            if (channelId == 1 && iter_times_ == dataset()->maxEvalSteps_) {
                *end_of_sequence = true;
            }

            int getNextStatus = GET_NEXT_CONTINUE;
            if (*end_of_sequence) {
                getNextStatus = GET_NEXT_TERMINATE;

                MPI_Request req;
                MPI_Iallreduce(MPI_IN_PLACE, &getNextStatus, 1, MPI_INT, MPI_SUM, g_comm[channelId],
                               &req);
                CheckCommFinished(req, channelId);

                keyProcess->EnqueueEosBatch(iter_times_, dataset()->channelId_);

                LOG_DEBUG("[ACTIVE] GetNext eos was triggered actively, channel: {}, iter: {}",
                          dataset()->channelId_,
                          iter_times_);

                input_impl_.reset();
                return Status::OK();
            }

            MPI_Request req;
            MPI_Iallreduce(MPI_IN_PLACE, &getNextStatus, 1, MPI_INT, MPI_SUM, g_comm[channelId], &req);
            CheckCommFinished(req, channelId);

            if (getNextStatus < g_rankSize) {
                *end_of_sequence = true;

                keyProcess->EnqueueEosBatch(iter_times_, dataset()->channelId_);

                LOG_DEBUG(
                    "[PASSIVE] GetNext eos was triggered passively, channel: {}, iter: {}, sum: {}",
                    dataset()->channelId_, iter_times_, getNextStatus);

                input_impl_.reset();
                return Status::OK();
            }

            iter_times_ += 1;
            return Status::OK();
        }

    protected:
        std::shared_ptr <model::Node> CreateNode(
                IteratorContext *ctx, model::Node::Args args) const override
                {
            return model::MakeKnownRatioNode(std::move(args), 1); // ratio = 1
        }

#if defined(TF_VERSION_TF2)
        Status SaveInternal(SerializationContext* ctx, IteratorStateWriter* writer) override
        {
            TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
            return Status::OK();
        }
#else

        Status SaveInternal(IteratorStateWriter *writer) override
        {
            TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
            return Status::OK();
        }

#endif

        Status RestoreInternal(IteratorContext *ctx,
                               IteratorStateReader *reader) override
                               {
            mutex_lock l(mu_);
            TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
            return Status::OK();
        }

        void PrintOutput(std::vector <Tensor> *out_tensors, int channelId)
        {
            // Out size equals to zero when batch eos.
            int outSize = out_tensors->size();
            if (MxRec::Logger::GetLevel() <= MxRec::Logger::DEBUG) {
                for (const auto& t : *out_tensors) {
                    DataType tensor_type = t.dtype();
                    TensorShape tensor_shape = t.shape();
                    LOG_DEBUG("Iterator getNext normal, channel: {}, iter: {}, outTensor size: {}, "
                              "tensor_type: {}, tensor_shape: {}",
                              channelId,
                              iter_times_,
                              outSize,
                              tensor_type,
                              tensor_shape.DebugString());
                }
            }
            if (outSize <= 0) {
                LOG_DEBUG("Iterator getNext eos, channel: {}, iter: {}, outTensor size: {}", channelId,
                          iter_times_, outSize);
            }
        }

    private:
        static constexpr int GET_NEXT_CONTINUE = 1;
        static constexpr int GET_NEXT_TERMINATE = 0;

        tensorflow::mutex mu_;
        int64 i_
        GUARDED_BY(mu_);
        int64 iter_times_
        GUARDED_BY(mu_);
        std::unique_ptr <IteratorBase> input_impl_
        GUARDED_BY(mu_);
    };

    const DatasetBase *input_;
    int32_t channelId_;
    int32_t maxTrainSteps_;
    int32_t maxEvalSteps_;
    const DataTypeVector outputTypes_;
    std::vector <PartialTensorShape> outputShapes_;
    int id_;
};

EosDatasetOp::EosDatasetOp(OpKernelConstruction *ctx) : UnaryDatasetOpKernel(ctx)
{
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &outputTypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &outputShapes_));
}

void EosDatasetOp::MakeDataset(OpKernelContext *ctx, DatasetBase *input, DatasetBase **output)
{
    int32_t channel;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int32_t>(ctx, kChannelId, &channel));
    int32_t maxTrainSteps;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int32_t>(ctx, kMaxTrainSteps, &maxTrainSteps));
    int32_t maxEvalSteps;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int32_t>(ctx, kMaxEvalSteps, &maxEvalSteps));
    *output = new (std::nothrow) Dataset(ctx, input, channel, maxTrainSteps, maxEvalSteps, outputTypes_, outputShapes_);
    OP_REQUIRES(ctx, *output != nullptr, errors::InvalidArgument("EosDatasetOp: new dataset failed"));
}

REGISTER_OP("EosDataset")
.Input("input_dataset: variant")
.Input("channel_id: int32")
.Input("max_train_steps: int32")
.Input("max_eval_steps: int32")
.Output("handle: variant")
.Attr("output_types: list(type) >= 1")
.Attr("output_shapes: list(shape) >= 1")
.SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("EosDataset").Device(DEVICE_CPU),
                        EosDatasetOp);

}  // namespace data
}  // namespace tensorflow
