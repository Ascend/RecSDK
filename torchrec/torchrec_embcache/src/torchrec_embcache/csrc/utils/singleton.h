/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_UTILS_SINGLETON_H
#define EMBEDDING_CACHE_UTILS_SINGLETON_H

#include <mutex>
#include <iostream>

namespace Embcache {

template <typename T>
class Singleton {
public:
    Singleton() = delete;
    Singleton& operator=(const Singleton& singleton) = delete;

    static T* GetInstance()
    {
        try {
            static T instance;
            return &instance;
        } catch (std::exception& e) {
            std::cout << "create singleton error..." << std::endl;
            return nullptr;
        }
    }
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_UTILS_SINGLETON_H