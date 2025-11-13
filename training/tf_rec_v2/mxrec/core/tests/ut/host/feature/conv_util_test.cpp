/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexcept>

#include "gtest/gtest.h"
#include "tensorflow/core/framework/tensor.h"

#include "feature/conv_util.h"
#include "common/types.h"

namespace rec_sdk {
namespace feature {

using common::emb_key_t;

TEST(ConvertTensorToVectorTest, NonEmptyTensor)
{
    tensorflow::Tensor tensor(tensorflow::DT_INT64, tensorflow::TensorShape({3}));
    auto tensorData = tensor.flat<emb_key_t>().data();
    tensorData[0] = 1;
    tensorData[1] = 2;
    tensorData[2] = 3;

    auto vec = ConvertTensorToVector1D<emb_key_t>(tensor);
    ASSERT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
}

TEST(ConvertTensorToVectorTest, EmptyTensor)
{
    tensorflow::Tensor tensor(tensorflow::DT_INT64, tensorflow::TensorShape({0}));
    EXPECT_THROW(ConvertTensorToVector1D<emb_key_t>(tensor), std::invalid_argument);
}

TEST(ConvertTensorToVectorTest, TensorDimsFail)
{
    tensorflow::Tensor tensor(tensorflow::DT_INT64, tensorflow::TensorShape({2, 3}));
    EXPECT_THROW(ConvertTensorToVector1D<emb_key_t>(tensor), std::invalid_argument);
}

TEST(ConvertVectorToTensorTest, NonEmptyVector)
{
    std::vector<emb_key_t> vec = {1, 2, 3};
    tensorflow::Tensor tensor = ConvertVectorToTensor1D(vec);

    ASSERT_EQ(tensor.NumElements(), 3);
    auto tensorData = tensor.flat<emb_key_t>().data();
    EXPECT_EQ(tensorData[0], 1);
    EXPECT_EQ(tensorData[1], 2);
    EXPECT_EQ(tensorData[2], 3);
}

TEST(ConvertVectorToTensorTest, EmptyVector)
{
    std::vector<emb_key_t> vec;
    EXPECT_THROW(ConvertVectorToTensor1D(vec), std::invalid_argument);
}

}  // namespace util
}  // namespace rec_sdk
