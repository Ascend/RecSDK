/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#ifndef MX_REC_HOSTEMB_H
#define MX_REC_HOSTEMB_H

#include <thread>
#include <vector>
#include <memory>
#include <array>
#include "absl/container/flat_hash_map.h"
#include "utils/common.h"
#include "utils/singleton.h"
#include "tensorflow/core/framework/tensor.h"

namespace MxRec {
    using namespace std;
    using namespace tensorflow;

    class HostEmb {
    public:
        HostEmb() = default;

        ~HostEmb()
        {};

        void Initialize(const vector<EmbInfo>& embInfos, int seed);

        void Join(int channelId);

        void UpdateEmb(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName);

        void UpdateEmbV2(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName);

        void GetH2DEmb(const vector<size_t>& missingKeysHostPos, const string& embName,
                       vector<Tensor>& h2dEmbOut);
        auto GetHostEmbs() -> absl::flat_hash_map<string, HostEmbTable>*;

        void EvictInitEmb(const string& embName, const vector<size_t>& offset);

        HostEmbTable& GetEmb(const string& embName)
        {
            return hostEmbs.at(embName);
        }

    GTEST_PRIVATE:
        absl::flat_hash_map<string, HostEmbTable> hostEmbs;

        std::vector<unique_ptr<std::thread>> procThreadsForTrain;
        std::vector<unique_ptr<std::thread>> procThreadsForEval;

        void EmbDataGenerator(const vector<InitializeInfo>& initializeInfos, int seed, int vocabSize, int embeddingSize,
                              vector<vector<float>>& embData) const;
        void EmbPartGenerator(const vector<InitializeInfo> &initializeInfos, vector<vector<float>> &embData,
                              const vector<size_t>& offset) const;
    };
}

#endif // MX_REC_HOSTEMB_H