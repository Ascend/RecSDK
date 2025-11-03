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

#ifndef UNIQUE_OCK_CTR_COMMON_H
#define UNIQUE_OCK_CTR_COMMON_H

#include <cstdint>
#include <string>
#include <memory>
#include "unique.h"
#include "embedding_cache.h"


#ifdef __cplusplus
extern "C" {
#endif

using ExternalLog = void (*)(int level, const char *msg);

#ifdef __cplusplus
}
#endif

#include "ock_ctr_common_def.h"

namespace ock {
namespace ctr {
class Factory;

using FactoryPtr = std::shared_ptr<Factory>;
using UniquePtr = std::shared_ptr<Unique>;
using EmbCacheManagerPtr = std::shared_ptr<EmbCache::EmbCacheManager>;

class Factory {
public:
    virtual ~Factory() = default;
    virtual int CreateUnique(UniquePtr &out) = 0;
    virtual int CreateEmbCacheManager(EmbCacheManagerPtr &out) = 0;
    virtual int SetExternalLogFuncInner(ExternalLog logFunc) = 0;

public:
    static int Create(FactoryPtr &out)
    {
        int result = 0;
        uintptr_t factory = 0;
        /* dynamic load function */
        if ((result = OckCtrCommonDef::CreateFactory(factory)) == 0) {
            out.reset(reinterpret_cast<Factory *>(factory));
        }
        return result;
    }
};
}
}

#endif // UNIQUE_OCK_CTR_COMMON_H
