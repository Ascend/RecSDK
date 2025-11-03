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

#include "factory_impl.h"

namespace ock {
namespace ctr {
Factory *FactoryImpl::gGlobalFactory = nullptr;
Lock FactoryImpl::gLock;

#ifdef __cplusplus
extern "C" {
#endif

int CTR_CreateFactory(uintptr_t& outFactory)
{
    if (FactoryImpl::gGlobalFactory == nullptr) {
        Locker<Lock> locker(&FactoryImpl::gLock);
        if (FactoryImpl::gGlobalFactory == nullptr) {
            auto tmp = new (std::nothrow) FactoryImpl();
            if (tmp == nullptr) {
                return H_NEW_OBJECT_FAILED;
            }

            FactoryImpl::gGlobalFactory = tmp;
        }
    }
    outFactory = reinterpret_cast<uintptr_t>(FactoryImpl::gGlobalFactory);
    return H_OK;
}
#ifdef __cplusplus
}
#endif

int FactoryImpl::CreateUnique(std::shared_ptr<Unique> &out)
{
    auto tmp = new (std::nothrow) UniqueImpl();
    if (tmp == nullptr) {
        return H_NEW_OBJECT_FAILED;
    }

    out.reset(dynamic_cast<Unique *>(tmp));
    return H_OK;
}

int FactoryImpl::CreateEmbCacheManager(std::shared_ptr<EmbCache::EmbCacheManager> &out)
{
    auto tmp = new (std::nothrow) EmbCache::EmbCacheManagerImpl();
    if (tmp == nullptr) {
        return H_NEW_OBJECT_FAILED;
    }

    out.reset(dynamic_cast<EmbCache::EmbCacheManager *>(tmp));
    return H_OK;
}

int FactoryImpl::SetExternalLogFuncInner(ExternalLog logFunc)
{
    auto logger = ExternalLogger::Instance();
    if (logger == nullptr) {
        std::cout << "Failed to create logger instance" << std::endl;
        return H_NEW_OBJECT_FAILED;
    }

    logger->SetExternalLogFunction(logFunc);
    return H_OK;
}
}
}
