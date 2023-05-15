/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:  singleton module.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef RC_UTILS_SINGLETON_H
#define RC_UTILS_SINGLETON_H


#include <mutex>
#include <iostream>

/**
 * T must be destructed
 * @tparam T
 */
template<typename T>
class Singleton {
public:
    Singleton() = delete;

    Singleton(const Singleton& singleton) = delete;

    Singleton& operator=(const Singleton& singleton) = delete;

    static T* GetInstance()
    {
        try {
            static T instance;
            return &instance;
        } catch (std::exception& e) {
            std::cerr << " create singleton error" << std::endl;
            return nullptr;
        }
    }

    template<typename... P>
    static T* GetInstance(P&& ... args)
    {
        try {
            static T instance(std::forward<P>(args)...);
            return &instance;
        } catch (std::exception& e) {
            std::cerr << " create singleton error" << std::endl;
            return nullptr;
        }
    }
};

#endif
