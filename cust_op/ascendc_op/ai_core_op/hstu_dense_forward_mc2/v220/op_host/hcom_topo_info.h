/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_
#define METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_

#include <unordered_map>
#include <mutex>

using Status = int32_t;

namespace ge {
constexpr uint32_t COMM_MESH = 0b1U;
constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
class HcomTopoInfo {
public:
    enum class TopoLevel {
        L0 = 0,
        L1,
        MAX,
    };
    struct TopoLevelDesc {
        uint32_t comm_sets;
        uint32_t rank_size;
    };
    using TopoDescs = TopoLevelDesc[static_cast<int32_t>(TopoLevel::MAX)];
    struct TopoInfo {
        int64_t rank_size;
        void* notify_handle;
        TopoDescs topo_level_descs;
    };
    static HcomTopoInfo& Instance();
    bool TopoInfoHasBeenSet(const char_t* group) const;
    bool TryGetGroupTopoInfo(const char_t* group, TopoInfo& info) const;
    Status SetGroupTopoInfo(const char_t* group, const TopoInfo& info);
    Status GetGroupRankSize(const char_t* group, int64_t& rank_size) const;
    TopoDescs* GetGroupTopoDesc(const char_t* group) const;
    Status GetGroupNotifyHandle(const char_t* group, void*& notify_handle) const;
    void UnsetGroupTopoInfo(const char_t* group)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        (void)rank_info_.erase(group);
    }

    Status SetGroupOrderedStream(const char_t* group, void* stream);
    Status GetGroupOrderedStream(const char_t* group, void*& stream) const;
    void UnsetGroupOrderedStream(const char_t* group)
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        (void)group_to_ordered_stream_.erase(group);
    }

    Status SetGroupOrderedStream(const int32_t device_id, const char_t* group, void* stream);
    Status GetGroupOrderedStream(const int32_t device_id, const char_t* group, void*& stream) const;
    void UnsetGroupOrderedStream(const int32_t device_id, const char_t* group);

private:
    HcomTopoInfo() = default;
    ~HcomTopoInfo() = default;
    std::unordered_map<std::string, TopoInfo> rank_info_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, void*> group_to_ordered_stream_;  // Ordered stream per communication group
    std::unordered_map<int32_t, std::unordered_map<std::string, void*>>
        device_id_to_group_to_ordered_stream_;  // Ordered stream per device + group
};
}  // namespace ge

#endif  // METADEF_CXX_INC_EXTERNAL_HCOM_HCOM_TOPO_INFO_H_
