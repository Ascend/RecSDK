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

#include <unordered_map>

#include "types.h"

namespace rec_sdk {
namespace common {

tensorflow::DataType MapDataTypeFromAclToTF(const aclDataType aclDType)
{
    static const std::unordered_map<aclDataType, tensorflow::DataType> aclToTfMap = {
        {ACL_FLOAT16, tensorflow::DT_HALF},  {ACL_FLOAT, tensorflow::DT_FLOAT},   {ACL_DOUBLE, tensorflow::DT_DOUBLE},
        {ACL_INT8, tensorflow::DT_INT8},     {ACL_INT16, tensorflow::DT_INT16},   {ACL_INT32, tensorflow::DT_INT32},
        {ACL_INT64, tensorflow::DT_INT64},   {ACL_UINT8, tensorflow::DT_UINT8},   {ACL_UINT16, tensorflow::DT_UINT16},
        {ACL_UINT32, tensorflow::DT_UINT32}, {ACL_UINT64, tensorflow::DT_UINT64}, {ACL_BOOL, tensorflow::DT_BOOL}};

    auto it = aclToTfMap.find(aclDType);
    if (it == aclToTfMap.end()) {
        return tensorflow::DT_INVALID;
    }

    return it->second;
}

aclDataType MapDataTypeFromTFToAcl(const tensorflow::DataType tfDType)
{
    static const std::unordered_map<tensorflow::DataType, aclDataType> tfToAclMap = {
        {tensorflow::DT_HALF, ACL_FLOAT16},  {tensorflow::DT_FLOAT, ACL_FLOAT},   {tensorflow::DT_DOUBLE, ACL_DOUBLE},
        {tensorflow::DT_INT8, ACL_INT8},     {tensorflow::DT_INT16, ACL_INT16},   {tensorflow::DT_INT32, ACL_INT32},
        {tensorflow::DT_INT64, ACL_INT64},   {tensorflow::DT_UINT8, ACL_UINT8},   {tensorflow::DT_UINT16, ACL_UINT16},
        {tensorflow::DT_UINT32, ACL_UINT32}, {tensorflow::DT_UINT64, ACL_UINT64}, {tensorflow::DT_BOOL, ACL_BOOL}};

    auto it = tfToAclMap.find(tfDType);
    if (it == tfToAclMap.end()) {
        return ACL_DT_UNDEFINED;
    }

    return it->second;
}

std::string GetAclTensorTypeStr(const acltdtTensorType tensorType)
{
    static const std::unordered_map<acltdtTensorType, std::string> tensorTypeMap = {
        {ACL_TENSOR_DATA_TENSOR, "ACL_TENSOR_DATA_TENSOR"},
        {ACL_TENSOR_DATA_SLICE_TENSOR, "ACL_TENSOR_DATA_SLICE_TENSOR"},
        {ACL_TENSOR_DATA_ABNORMAL, "ACL_TENSOR_DATA_ABNORMAL"},
        {ACL_TENSOR_DATA_END_TENSOR, "ACL_TENSOR_DATA_END_TENSOR"},
        {ACL_TENSOR_DATA_END_OF_SEQUENCE, "ACL_TENSOR_DATA_END_OF_SEQUENCE"}};

    auto it = tensorTypeMap.find(tensorType);
    if (it == tensorTypeMap.end()) {
        return "ACL_TENSOR_DATA_UNDEFINED";
    }

    return it->second;
}

std::string GetAclDataTypeStr(const aclDataType dataType)
{
    static const std::unordered_map<aclDataType, std::string> dataTypeMap = {
        {ACL_FLOAT16, "ACL_FLOAT16"}, {ACL_FLOAT, "ACL_FLOAT"},   {ACL_DOUBLE, "ACL_DOUBLE"}, {ACL_INT8, "ACL_INT8"},
        {ACL_INT16, "ACL_INT16"},     {ACL_INT32, "ACL_INT32"},   {ACL_INT64, "ACL_INT64"},   {ACL_UINT8, "ACL_UINT8"},
        {ACL_UINT16, "ACL_UINT16"},   {ACL_UINT32, "ACL_UINT32"}, {ACL_UINT64, "ACL_UINT64"}, {ACL_BOOL, "ACL_BOOL"}};

    auto it = dataTypeMap.find(dataType);
    if (it == dataTypeMap.end()) {
        return "ACL_DT_UNDEFINED";
    }

    return it->second;
}

}  // namespace common
}  // namespace rec_sdk
