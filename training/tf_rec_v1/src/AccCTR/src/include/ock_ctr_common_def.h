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

#ifndef OCK_OCK_CTR_COMMON_DEF_H
#define OCK_OCK_CTR_COMMON_DEF_H

#include <dlfcn.h>
#include <iostream>
#include <mutex>

using CTR_CREATE_FACTORY_FUNCTION = int (*)(uintptr_t *);

namespace ock {
namespace ctr {
class OckCtrCommonDef {
public:
    static int CreateFactory(uintptr_t *factory)
    {
        static void *handle = nullptr;
        static std::mutex m;
        std::unique_lock<std::mutex> lock(m);
        if (handle != nullptr) {
            std::cout << "can't create factory more than 1 time." << std::endl;
            return -1;
        }

        handle = dlopen(LIBRARY_NAME, RTLD_NOW);
        if (handle == nullptr) {
            std::cout << "Failed to call dlopen to load library '" << LIBRARY_NAME << "', error " << dlerror() <<
                std::endl;
            return -1;
        }

        auto fun = (CTR_CREATE_FACTORY_FUNCTION)dlsym(handle, "CTR_CreateFactory");
        if (fun == nullptr) {
            std::cout << "Failed to call dlsym to load function 'CTR_CreateFactory', error " << dlerror() << std::endl;
            dlclose(handle);
            return -1;
        }

        return fun(factory);
    }

private:
    constexpr static const char *LIBRARY_NAME = "lib_ock_ctr_common.so";
};
}
}

#endif // OCK_OCK_CTR_COMMON_DEF_H
