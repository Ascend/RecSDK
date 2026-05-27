/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef DYNAMIC_EMB_TABLE_VECTOR_H_
#define DYNAMIC_EMB_TABLE_VECTOR_H_

#include <cstdint>
#include <simt_api/common_functions.h>
#include "kernel_operator.h"

namespace dyn_emb {

template <typename ElementType>
struct TableVector {
    struct Args {
        __gm__ ElementType* __gm__* vec_ptrs{nullptr};
        __gm__ bool* founds{nullptr};
    };

    __forceinline__ __simt_callee__ TableVector(Args args)
        : vec_ptrs_(args.vec_ptrs),
          founds_(args.founds),
          vec_id_(-1),
          vec_ptr_(nullptr),
          found_(false)
    {
    }

    __forceinline__ __simt_callee__ bool isInitialized(int64_t vec_id)
    {
        if (vec_id != vec_id_) {
            load(vec_id);
        }
        return found_;
    }

    __forceinline__ __simt_callee__ bool isValid(int64_t vec_id)
    {
        if (vec_id != vec_id_) {
            load(vec_id);
        }
        return vec_ptr_ != nullptr;
    }

    __forceinline__ __simt_callee__ __gm__ ElementType* data_ptr(int64_t vec_id, int i = 0)
    {
        if (vec_id != vec_id_) {
            load(vec_id);
        }
        if (vec_ptr_ != nullptr) {
            return vec_ptr_ + i;
        } else {
            return nullptr;
        }
    }

private:
    __forceinline__ __simt_callee__ void load(int64_t vec_id)
    {
        vec_id_ = vec_id;
        found_ = founds_[vec_id];
        vec_ptr_ = vec_ptrs_[vec_id];
    }

    __gm__ ElementType* __gm__* vec_ptrs_;
    __gm__ bool* founds_;
    int64_t vec_id_;
    __gm__ ElementType* vec_ptr_;
    bool found_;
};

template <typename T>
struct TableVectorSimd {
    struct Args {
        __gm__ T* __gm__* vec_ptrs{nullptr};
        __gm__ bool* founds{nullptr};
    };

    __forceinline__ __aicore__ void init_simd(Args args)
    {
        vec_ptrs_ = args.vec_ptrs;
        founds_ = args.founds;
        vec_id_ = -1;
        vec_ptr_ = nullptr;
        found_ = false;
    }

    __forceinline__ __aicore__ bool isInitialized(int64_t vec_id)
    {
        if (vec_id != vec_id_) {
            load(vec_id);
        }
        return found_;
    }

    __forceinline__ __aicore__ bool isValid(int64_t vec_id)
    {
        if (vec_id != vec_id_) {
            load(vec_id);
        }
        return vec_ptr_ != nullptr;
    }

    __forceinline__ __aicore__ __gm__ T* data_ptr(int64_t vec_id, int i = 0)
    {
        if (vec_id != vec_id_) {
            load(vec_id);
        }
        if (vec_ptr_ != nullptr) {
            return vec_ptr_ + i;
        } else {
            return nullptr;
        }
    }

private:
    __forceinline__ __aicore__ void load(int64_t vec_id)
    {
        vec_id_ = vec_id;
        found_ = founds_[vec_id];
        vec_ptr_ = vec_ptrs_[vec_id];
    }

    __gm__ T* __gm__* vec_ptrs_;
    __gm__ bool* founds_;
    int64_t vec_id_;
    __gm__ T* vec_ptr_;
    bool found_;
};

}  // namespace dyn_emb

#endif  // DYNAMIC_EMB_TABLE_VECTOR_H_
