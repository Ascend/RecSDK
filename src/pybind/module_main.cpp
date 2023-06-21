/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: pybind module
 * Author: MindX SDK
 * Date: 2022/11/15
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dcmi_interface_api.h>

#include "hybrid_mgmt/hybrid_mgmt.h"
#include "module_main.h"

namespace py = pybind11;
using namespace MxRec;

void GetRankInfo(py::module_& m);

void GetEmbInfo(py::module_& m);

void GetRandomInfo(py::module_& m);

void GetHybridMgmt(py::module_& m);

void GetThresholdValue(pybind11::module_& m);

void GetInitializeInfo(pybind11::module_& m);

void GetConstantInitializerInfo(pybind11::module_& m);

void GetNormalInitializerInfo(pybind11::module_& m);

int GetUBHotSize(int devID)
{
    return static_cast<int>(static_cast<float>(MxRec::GetUBSize(devID)) / sizeof(float) * HOT_EMB_CACHE_PCT) ;
}

uint32_t GetLogicID(uint32_t phyid)
{
    int32_t ret = 0;
    uint32_t logicId;
    ret = dcmi_get_device_logicid_from_phyid(phyid, &logicId);
    if (ret != 0) {
        return ret;
    }
    return logicId;
}

PYBIND11_MODULE(mxrec_pybind, m)
{
    m.def("get_ub_hot_size", &GetUBHotSize, py::arg("device_id"));

    m.def("get_logic_id", &GetLogicID, py::arg("physic_id"));

    m.attr("USE_STATIC") = py::int_(HybridOption::USE_STATIC);

    m.attr("USE_HOT") = py::int_(HybridOption::USE_HOT);

    m.attr("USE_DYNAMIC_EXPANSION") = py::int_(HybridOption::USE_DYNAMIC_EXPANSION);

    GetRankInfo(m);

    GetEmbInfo(m);

    GetRandomInfo(m);

    GetHybridMgmt(m);

    GetThresholdValue(m);

    GetInitializeInfo(m);

    GetConstantInitializerInfo(m);

    GetNormalInitializerInfo(m);
}

void GetRankInfo(pybind11::module_& m)
{
    pybind11::class_<RankInfo>(m, "RankInfo")
        .def(py::init<int, int, int, int, int, vector<int>>(), py::arg("rank_id"), py::arg("device_id"),
             py::arg("local_rank_size"), py::arg("option"), py::arg("num_batch") = 1,
             py::arg("max_step") = vector<int> { -1, -1 })
        .def_readwrite("rank_id", &RankInfo::rankId)
        .def_readwrite("device_id", &RankInfo::deviceId)
        .def_readwrite("rank_size", &RankInfo::rankSize)
        .def_readwrite("local_rank_size", &RankInfo::localRankSize)
        .def_readwrite("option", &RankInfo::option)
        .def_readwrite("num_batch", &RankInfo::nBatch)
        .def_readwrite("max_step", &RankInfo::maxStep);
}

void GetEmbInfo(pybind11::module_& m)
{
    pybind11::class_<EmbInfo>(m, "EmbInfo")
            .def(pybind11::init<const std::string&, int, int, int, bool, bool, std::vector<string>, std::vector<size_t>,
                    std::vector<InitializeInfo>&, std::map<std::string, int>>(),
                 py::arg("name"), py::arg("send_count"), py::arg("embedding_size"),
                 py::arg("ext_embedding_size"), py::arg("modify_graph"),
                 py::arg("is_save"), py::arg("channel_name_list"),
                 py::arg("vocab_size"), py::arg("initialize_infos"), py::arg("send_count_map"))
            .def_readwrite("name", &EmbInfo::name)
            .def_readwrite("send_count", &EmbInfo::sendCount)
            .def_readwrite("embedding_size", &EmbInfo::embeddingSize)
            .def_readwrite("ext_embedding_size", &EmbInfo::extEmbeddingSize)
            .def_readwrite("modify_graph", &EmbInfo::modifyGraph)
            .def_readwrite("is_save", &EmbInfo::isSave)
            .def_readwrite("channel_name_list", &EmbInfo::channelNames)
            .def_readwrite("dev_vocab_size", &EmbInfo::devVocabSize)
            .def_readwrite("host_vocab_size", &EmbInfo::hostVocabSize)
            .def_readwrite("initialize_infos", &EmbInfo::initializeInfos)
            .def_readwrite("send_count_map", &EmbInfo::sendCountMap);
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

void GetInitializeInfo(pybind11::module_ &m)
{
    pybind11::class_<InitializeInfo>(m, "InitializeInfo")
        .def(py::init<std::string &, int, int, ConstantInitializerInfo>(), py::arg("name"), py::arg("start"),
        py::arg("len"), py::arg("constant_initializer_info"))
        .def(py::init<std::string &, int, int, NormalInitializerInfo>(), py::arg("name"), py::arg("start"),
        py::arg("len"), py::arg("normal_initializer_info"))
        .def_readwrite("name", &InitializeInfo::name)
        .def_readwrite("start", &InitializeInfo::start)
        .def_readwrite("len", &InitializeInfo::len)
        .def_readwrite("ConstantInitializerInfo", &InitializeInfo::constantInitializerInfo)
        .def_readwrite("NormalInitializerInfo", &InitializeInfo::normalInitializerInfo);
}

void GetConstantInitializerInfo(pybind11::module_ &m)
{
    pybind11::class_<ConstantInitializerInfo>(m, "ConstantInitializerInfo")
        .def(py::init<float, float>(), py::arg("constant_val") = 0, py::arg("initK") = 1.0)
        .def_readwrite("constant_val", &ConstantInitializerInfo::constantValue)
        .def_readwrite("initK", &ConstantInitializerInfo::initK);
}

void GetNormalInitializerInfo(pybind11::module_ &m)
{
    pybind11::class_<NormalInitializerInfo>(m, "NormalInitializerInfo")
        .def(py::init<float, float, int, float>(), py::arg("mean") = 0.0, py::arg("stddev") = 1.0, py::arg("seed") = 0,
             py::arg("initK") = 1.0)
        .def_readwrite("mean", &NormalInitializerInfo::mean)
        .def_readwrite("stddev", &NormalInitializerInfo::stddev)
        .def_readwrite("seed", &NormalInitializerInfo::seed)
        .def_readwrite("initK", &NormalInitializerInfo::initK);
}

void GetHybridMgmt(pybind11::module_& m)
{
    pybind11::class_<HybridMgmt>(m, "HybridMgmt")
        .def(py::init())
        .def("initialize", &MxRec::HybridMgmt::Initialize, py::arg("rank_info"), py::arg("emb_info"),
        py::arg("seed") = DEFAULT_RANDOM_SEED, py::arg("threshold_values") = vector<ThresholdValue> {},
        py::arg("if_load") = false)
        .def("save", &MxRec::HybridMgmt::Save, py::arg("save_path") = "")
        .def("load", &MxRec::HybridMgmt::Load, py::arg("load_path") = "")
        .def("destroy", &MxRec::HybridMgmt::Destroy)
        .def("evict", &MxRec::HybridMgmt::Evict);
}

void GetThresholdValue(pybind11::module_& m)
{
    pybind11::class_<ThresholdValue>(m, "ThresholdValue")
        .def(pybind11::init<string, int, int>())
        .def_readwrite("tensor_name", &ThresholdValue::tensorName)
        .def_readwrite("count_threshold", &ThresholdValue::countThreshold)
        .def_readwrite("time_threshold", &ThresholdValue::timeThreshold);
}
