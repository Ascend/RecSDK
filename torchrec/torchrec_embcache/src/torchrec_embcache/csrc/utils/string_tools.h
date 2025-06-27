/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_UTILS_STRING_TOOLS_H
#define EMBEDDING_CACHE_UTILS_STRING_TOOLS_H

#include <vector>
#include <string>
#include <sstream>

namespace Embcache {

class StringTools {
public:
    template <typename K, typename V>
    static std::string ToString(std::vector<std::pair<K, V>>& items)
    {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < items.size(); i++) {
            ss << items[i].first << ":" << items[i].second;
            if (i != items.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    template <typename T>
    static std::string ToString(const std::vector<T>& items)
    {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < items.size(); i++) {
            ss << items[i];
            if (i != items.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    template <typename T>
    static std::string ToString(const T* data, size_t size)
    {
        if (data == nullptr || size == 0) {
            return "[]";
        }

        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < size; ++i) {
            if (i != 0) {
                ss << ", ";
            }
            ss << data[i];
        }
        ss << "]";

        return ss.str();
    }
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_UTILS_STRING_TOOLS_H