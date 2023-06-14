/*
 * @Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * @Description:
 * @Version: 1.0
 * @Author: dev
 * @Date: 2023-05-5 09:50:00
 * @LastEditors: dev
 * @LastEditTime: 2023-05-5 09:50:00
 */

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
    static int CreatFactory(uintptr_t *factory)
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

        fun(factory);
        return 0;
    }

private:
    constexpr static const char *LIBRARY_NAME = "lib_ock_ctr_common.so";
};
}
}

#endif // OCK_OCK_CTR_COMMON_DEF_H
