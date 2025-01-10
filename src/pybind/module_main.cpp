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

#include <dsmi_common_interface.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lcal_comm.h"
#include "hybrid_mgmt/hybrid_mgmt.h"

namespace py = pybind11;
using namespace MxRec;
namespace {
void GetRankInfo(py::module_& m);

void GetEmbInfoParams(py::module_& m);

void GetEmbInfo(py::module_& m);

void GetRandomInfo(py::module_& m);

void GetOptimizerInfo(py::module_& m);

void GetHybridMgmt(py::module_& m);

void GetThresholdValue(pybind11::module_& m);

void GetInitializeInfo(pybind11::module_& m);

void GetConstantInitializerInfo(pybind11::module_& m);

void GetNormalInitializerInfo(pybind11::module_& m);

int GetUBHotSize(int devID)
{
    return static_cast<int>(static_cast<float>(MxRec::GetUBSize(devID)) / sizeof(float) * HOT_EMB_CACHE_PCT);
}

int32_t GetLogicID(uint32_t phyid)
{
    uint32_t logicId;
    int32_t ret = dsmi_get_logicid_from_phyid(phyid, &logicId);
    if (ret != 0) {
        return ret;
    }
    return logicId;
}

uint32_t GetDeviceCount()
{
    uint32_t count;
    aclError ec = aclrtGetDeviceCount(&count);
    if (ec != 0) {
        throw runtime_error("The failed to get device count.");
    }
    return count;
}

vector<int64_t> GetPeerMem(int32_t rankId, int deviceId, int rankSize)
{
    static bool firstGetPeerMem = true;
    static constexpr int32_t maxRankPerNode = 16;
    int32_t localRankId = rankId % maxRankPerNode;
    auto ret = aclrtSetDevice(localRankId);
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Set device failed, device_id:{}", localRankId);
        return {};
    }
    auto tmp = vector<int>{deviceId};
    static Lcal::LcalComm c(rankId, rankSize, tmp);
    if (firstGetPeerMem) {
        c.Init();
    }
    firstGetPeerMem = false;
    static vector <int64_t> peerMem;
    for (int i = 0; i < rankSize; i++) {
        peerMem.emplace_back(reinterpret_cast<int64_t>(c.peerMem_[i]));
    }
    return peerMem;
}

PYBIND11_MODULE(mxrec_pybind, m)
{
    m.def("get_peer_mem", &GetPeerMem, py::arg("rank_id"), py::arg("device_id"), py::arg("rank_size"));

    m.def("get_ub_hot_size", &GetUBHotSize, py::arg("device_id"));

    m.def("get_logic_id", &GetLogicID, py::arg("physic_id"));

    m.def("get_device_count", &GetDeviceCount);

    m.attr("USE_STATIC") = py::int_(HybridOption::USE_STATIC);

    m.attr("USE_DYNAMIC_EXPANSION") = py::int_(HybridOption::USE_DYNAMIC_EXPANSION);

    m.attr("USE_SUM_SAME_ID_GRADIENTS") = py::int_(HybridOption::USE_SUM_SAME_ID_GRADIENTS);

    GetRankInfo(m);

    GetEmbInfoParams(m);

    GetEmbInfo(m);

    GetRandomInfo(m);

    GetOptimizerInfo(m);

    GetHybridMgmt(m);

    GetThresholdValue(m);

    GetInitializeInfo(m);

    GetConstantInitializerInfo(m);

    GetNormalInitializerInfo(m);
}

void GetRankInfo(pybind11::module_& m)
{
    pybind11::class_<RankInfo>(m, "RankInfo")
        .def(py::init<int, int, int, int, vector<int>>(), py::arg("rank_id"), py::arg("device_id"),
             py::arg("local_rank_size"), py::arg("option"), py::arg("ctrl_steps") = vector<int>{-1, -1, -1})
        .def_readwrite("rank_id", &RankInfo::rankId)
        .def_readwrite("device_id", &RankInfo::deviceId)
        .def_readwrite("rank_size", &RankInfo::rankSize)
        .def_readwrite("local_rank_size", &RankInfo::localRankSize)
        .def_readwrite("option", &RankInfo::option)
        .def_readwrite("ctrl_steps", &RankInfo::ctrlSteps);
}

void GetEmbInfoParams(pybind11::module_& m)
{
    pybind11::class_<EmbInfoParams>(m, "EmbInfoParams")
        .def(pybind11::init<const std::string&, int, int, int, bool, bool, bool, bool>(), py::arg("name"),
             py::arg("send_count"), py::arg("embedding_size"), py::arg("ext_embedding_size"), py::arg("is_save"),
             py::arg("is_grad"), py::arg("is_dp"), py::arg("padding_keys_mask"))
        .def_readwrite("name", &EmbInfoParams::name)
        .def_readwrite("send_count", &EmbInfoParams::sendCount)
        .def_readwrite("embedding_size", &EmbInfoParams::embeddingSize)
        .def_readwrite("ext_embedding_size", &EmbInfoParams::extEmbeddingSize)
        .def_readwrite("is_save", &EmbInfoParams::isSave)
        .def_readwrite("is_grad", &EmbInfoParams::isGrad)
        .def_readwrite("is_dp", &EmbInfoParams::isDp)
        .def_readwrite("padding_keys_mask", &EmbInfoParams::paddingKeysMask);
}

void GetEmbInfo(pybind11::module_& m)
{
    pybind11::class_<EmbInfo>(m, "EmbInfo")
        .def(pybind11::init<const EmbInfoParams&, std::vector<size_t>, std::vector<EmbCache::InitializerInfo>&,
                            std::vector<std::string>&, std::vector<emb_key_t>&>(),
             py::arg("embInfoParams"), py::arg("vocab_size"), py::arg("initialize_infos"), py::arg("ssd_data_path"),
             py::arg("padding_keys"))
        .def_readwrite("name", &EmbInfo::name)
        .def_readwrite("send_count", &EmbInfo::sendCount)
        .def_readwrite("embedding_size", &EmbInfo::embeddingSize)
        .def_readwrite("ext_embedding_size", &EmbInfo::extEmbeddingSize)
        .def_readwrite("is_save", &EmbInfo::isSave)
        .def_readwrite("is_grad", &EmbInfo::isGrad)
        .def_readwrite("is_dp", &EmbInfo::isDp)
        .def_readwrite("padding_keys_mask", &EmbInfo::paddingKeysMask)
        .def_readwrite("dev_vocab_size", &EmbInfo::devVocabSize)
        .def_readwrite("host_vocab_size", &EmbInfo::hostVocabSize)
        .def_readwrite("initialize_infos", &EmbInfo::initializeInfos)
        .def_readwrite("ssd_data_path", &EmbInfo::ssdDataPath)
        .def_readwrite("padding_keys", &EmbInfo::paddingKeys);
}

void GetRandomInfo(pybind11::module_& m)
{
    pybind11::class_<RandomInfo>(m, "RandomInfo")
        .def(pybind11::init<int, int, float, float, float>())
        .def_readwrite("start", &RandomInfo::start)
        .def_readwrite("len", &RandomInfo::len)
        .def_readwrite("constant_val", &RandomInfo::constantVal)
        .def_readwrite("random_min", &RandomInfo::randomMin)
        .def_readwrite("random_max", &RandomInfo::randomMax);
}

void GetOptimizerInfo(pybind11::module_& m)
{
    pybind11::class_<OptimizerInfo>(m, "OptimizerInfo")
        .def(pybind11::init<std::string, vector<std::string>>())
        .def_readwrite("name", &OptimizerInfo::optimName)
        .def_readwrite("params", &OptimizerInfo::optimParams);
}

void GetInitializeInfo(pybind11::module_& m)
{
    pybind11::class_<EmbCache::InitializerInfo>(m, "InitializeInfo")
        .def(py::init<std::string&, uint32_t, uint32_t, EmbCache::ConstantInitializerInfo>(), py::arg("name"),
             py::arg("start"), py::arg("len"), py::arg("constant_initializer_info"))
        .def(py::init<std::string&, uint32_t, uint32_t, EmbCache::NormalInitializerInfo>(), py::arg("name"),
             py::arg("start"), py::arg("len"), py::arg("normal_initializer_info"))
        .def_readwrite("name", &EmbCache::InitializerInfo::name)
        .def_readwrite("start", &EmbCache::InitializerInfo::start)
        .def_readwrite("len", &EmbCache::InitializerInfo::len)
        .def_readwrite("ConstantInitializerInfo", &EmbCache::InitializerInfo::constantInitializerInfo)
        .def_readwrite("NormalInitializerInfo", &EmbCache::InitializerInfo::normalInitializerInfo);
}

void GetConstantInitializerInfo(pybind11::module_& m)
{
    pybind11::class_<EmbCache::ConstantInitializerInfo>(m, "ConstantInitializerInfo")
        .def(py::init<float, float>(), py::arg("constant_val") = 0, py::arg("initK") = 1.0)
        .def_readwrite("constant_val", &EmbCache::ConstantInitializerInfo::constantValue)
        .def_readwrite("initK", &EmbCache::ConstantInitializerInfo::initK);
}

void GetNormalInitializerInfo(pybind11::module_& m)
{
    pybind11::class_<EmbCache::NormalInitializerInfo>(m, "NormalInitializerInfo")
        .def(py::init<float, float, uint32_t, float>(), py::arg("mean") = 0.0, py::arg("stddev") = 1.0,
             py::arg("seed") = 0, py::arg("initK") = 1.0)
        .def_readwrite("mean", &EmbCache::NormalInitializerInfo::mean)
        .def_readwrite("stddev", &EmbCache::NormalInitializerInfo::stddev)
        .def_readwrite("seed", &EmbCache::NormalInitializerInfo::seed)
        .def_readwrite("initK", &EmbCache::NormalInitializerInfo::initK);
}

void GetHybridMgmt(pybind11::module_& m)
{
    pybind11::class_<HybridMgmt>(m, "HybridMgmt")
        .def(py::init())
        .def("initialize", &MxRec::HybridMgmt::Initialize, py::arg("rank_info"), py::arg("emb_info"),
             py::arg("seed") = DEFAULT_RANDOM_SEED, py::arg("threshold_values") = vector<ThresholdValue>{},
             py::arg("if_load") = false, py::arg("is_incremental_checkpoint") = false)
        .def("save", &MxRec::HybridMgmt::Save, py::arg("save_path") = "", py::arg("save_delta") = false)
        .def("load", &MxRec::HybridMgmt::Load, py::arg("load_path") = "",
             py::arg("warm_start_tables") = vector<string>{})
        .def("destroy", &MxRec::HybridMgmt::Destroy)
        .def("evict", &MxRec::HybridMgmt::Evict)
        .def("fetch_device_emb", &MxRec::HybridMgmt::FetchDeviceEmb)
        .def("send", &MxRec::HybridMgmt::SendHostMap, py::arg("table_name") = "")
        .def("send_load_offset", &MxRec::HybridMgmt::SendLoadMap, py::arg("table_name") = "")
        .def("block_notify_wake", &MxRec::HybridMgmt::NotifyBySessionRun, py::arg("channel_id"))
        .def("block_count_steps", &MxRec::HybridMgmt::CountStepBySessionRun, py::arg("channel_id"),
             py::arg("steps") = 1)
        .def("get_table_size", &MxRec::HybridMgmt::GetTableSize, py::arg("table_name"))
        .def("get_table_capacity", &MxRec::HybridMgmt::GetTableCapacity, py::arg("table_name"))
        .def("set_optim_info", &MxRec::HybridMgmt::SetOptimizerInfo, py::arg("table_name"), py::arg("optimizer_info"))
        .def("start_sync_thread", &MxRec::HybridMgmt::StartSyncThread);
}

void GetThresholdValue(pybind11::module_& m)
{
    pybind11::class_<ThresholdValue>(m, "ThresholdValue")
        .def(pybind11::init<string, int, int, int, bool>())
        .def_readwrite("table_name", &ThresholdValue::tableName)
        .def_readwrite("count_threshold", &ThresholdValue::countThreshold)
        .def_readwrite("time_threshold", &ThresholdValue::timeThreshold)
        .def_readwrite("faae_coefficient", &ThresholdValue::faaeCoefficient)
        .def_readwrite("is_enable_sum", &ThresholdValue::isEnableSum);
}

}  // namespace
