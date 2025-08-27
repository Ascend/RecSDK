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

#ifndef RC_UTILS_SINGLETON_H
#define RC_UTILS_SINGLETON_H


#include <mutex>
#include <iostream>

/**
 * T must be destructed
 * @tparam T
 */
namespace MxRec {
    template<typename T>
    class Singleton {
    public:
        Singleton() = delete;

        Singleton(const Singleton &singleton) = delete;

        Singleton &operator=(const Singleton &singleton) = delete;

        static T *GetInstance()
        {
            try {
                static T instance;
                return &instance;
            } catch (std::exception &e) {
                std::cerr << " create singleton error" << std::endl;
                return nullptr;
            }
        }

        template<typename... P>
        static T *GetInstance(P &&... args)
        {
            try {
                static T instance(std::forward<P>(args)...);
                return &instance;
            } catch (std::exception &e) {
                std::cerr << " create singleton error" << std::endl;
                return nullptr;
            }
        }
    };
}
#endif
