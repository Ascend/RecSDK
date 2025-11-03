/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef LCCL_LCAL_API_H
#define LCCL_LCAL_API_H

#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

using LcalCommPtr = void*;
#define LCAL_UNIQUE_ID_BYTES 128
struct LcalUniqueId {
    char internal[LCAL_UNIQUE_ID_BYTES];
};

int LcalGetUniqueId(LcalUniqueId* uniqueId);

int LcalCommInitRankLocal(int rankSize, int rank, LcalCommPtr* comm);

int LcalCommInitRank(LcalUniqueId commId, int rankSize, int rank, LcalCommPtr* comm);

int LcalGetSharedMem(LcalCommPtr comm, int8_t*& commArgsPtr);

int LcalCommInit(int rank, int rankSize, LcalCommPtr* comms);

int LcalCommInitAll(uint32_t ndev, int32_t* devices, LcalCommPtr* comms);

int LcclCommDestroy(LcalCommPtr comm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // LCCL_LCAL_API_H
