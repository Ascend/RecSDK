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

#ifndef REC_SDK_TYPES_H
#define REC_SDK_TYPES_H

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

#include "acl/acl_base.h"
#include "acl/acl_tdt.h"

namespace rec_sdk {
namespace common {

using i32 = tensorflow::int32;
using i64 = tensorflow::int64;

using u32 = tensorflow::uint32;
using u64 = tensorflow::uint64;

using emb_key_t = i64;

tensorflow::DataType MapDataTypeFromAclToTF(const aclDataType aclDataType);

aclDataType MapDataTypeFromTFToAcl(const tensorflow::DataType tfDataType);

std::string GetAclTensorTypeStr(const acltdtTensorType tensorType);

std::string GetAclDataTypeStr(const aclDataType dataType);

template <typename T>
tensorflow::DataType GetTFDataType();

template <>
inline tensorflow::DataType GetTFDataType<i32>()
{
    return tensorflow::DT_INT32;
}

template <>
inline tensorflow::DataType GetTFDataType<i64>()
{
    return tensorflow::DT_INT64;
}

template <>
inline tensorflow::DataType GetTFDataType<u32>()
{
    return tensorflow::DT_UINT32;
}

template <>
inline tensorflow::DataType GetTFDataType<u64>()
{
    return tensorflow::DT_UINT64;
}

}  // namespace common
}  // namespace rec_sdk

#endif // REC_SDK_TYPES_H
