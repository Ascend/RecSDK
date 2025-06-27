/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_GLOGGER_H
#define EMBEDDING_CACHE_GLOGGER_H

#include <glog/logging.h>

namespace Embcache {

class Glogger {
public:
    void Init()
    {
        if (inited) {
            return;
        }

        google::InitGoogleLogging("embcache");
        FLAGS_logtostderr = true;

        // 0:INFO; 1:WARNING; 2:ERROR; 3:FATAL
        FLAGS_minloglevel = 1;
        char* minLogLevel = getenv("GLOG_MIN_LOG_LEVEL");
        if (minLogLevel) {
            FLAGS_minloglevel = atoi(minLogLevel);
        }

        inited = true;
    }

private:
    bool inited = false;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_GLOGGER_H