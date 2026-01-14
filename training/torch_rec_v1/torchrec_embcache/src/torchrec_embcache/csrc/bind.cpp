/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <any>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

#include "embedding_cache/embcache_manager.h"
#include "ops/restore.h"
#include "common/constants.h"

using namespace Embcache;

void AddEmbCacheManager(pybind11::module_& m)
{
    py::class_<EmbcacheManager>(m, "EmbcacheManager")
        .def(py::init<const std::vector<EmbConfig>&, bool>(), py::arg("emb_configs"),
             py::arg("need_accumulate_offset") = true)
        .def("compute_swap_info_async", &EmbcacheManager::ComputeSwapInfoAsync, py::arg("batch_keys"),
             py::arg("jagged_offs"), py::arg("table_indices") = std::vector<int32_t>{})
        .def("embedding_lookup_async", &EmbcacheManager::EmbeddingLookupAsync, py::arg("swap_info"),
             py::arg("table_indices") = std::vector<int32_t>{})
        .def("embedding_update_async", &EmbcacheManager::EmbeddingUpdateAsync, py::arg("swap_info"),
             py::arg("swapout_embs"), py::arg("swapout_optims"), py::arg("table_indices") = std::vector<int32_t>{})
        .def("record_timestamp", &EmbcacheManager::RecordTimestamp, py::arg("batch_keys"), py::arg("jagged_offs"),
             py::arg("batch_timestamps"), py::arg("table_indices") = std::vector<int32_t>{})
        .def("evict_features", &EmbcacheManager::EvictFeatures)
        .def("statistics_key_count", &EmbcacheManager::StatisticsKeyCount, py::arg("batch_keys"), py::arg("offset"),
             py::arg("batch_key_counts"), py::arg("table_index"))
        .def("record_embedding_update_times", &EmbcacheManager::RecordEmbeddingUpdateTimes)
        .def("record_batch_keys_async", &EmbcacheManager::RecordBatchKeysAsync, py::arg("batch_keys"),
             py::arg("jagged_offs"), py::arg("table_indices") = std::vector<int32_t>{})
        .def("embedding_to_host", &EmbcacheManager::Embedding2Host, py::arg("weights_dev"), py::arg("momentum_devs"))
        .def("save", &EmbcacheManager::Save, py::arg("path"), py::arg("rank"), py::arg("incremental"))
        .def("load", &EmbcacheManager::Load, py::arg("path"), py::arg("rank"), py::arg("incremental"));
}

void AddInitializerType(pybind11::module_& m)
{
    py::enum_<InitializerType>(m, "InitializerType")
        .value("LINEAR", InitializerType::LINEAR)
        .value("TRUNCATED_NORMAL", InitializerType::TRUNCATED_NORMAL)
        .value("UNIFORM", InitializerType::UNIFORM)
        .export_values();
}

void AddEmbConfigModule(pybind11::module_& m)
{
    pybind11::class_<EmbConfig>(m, "EmbConfig")
        .def(py::init<>())
        .def(pybind11::init<const std::string&, InitializerType, int32_t, int32_t, int64_t, float, float, float, float,
                            AdmitAndEvictConfig, int32_t, int32_t, int64_t, bool>(),
             py::arg("table_name"), py::arg("initializer_type"), py::arg("emb_dim"), py::arg("optim_num"),
             py::arg("cache_size"), py::arg("weight_init_min"), py::arg("weight_init_max"), py::arg("weight_init_mean"),
             py::arg("weight_init_stddev"), py::arg("admit_and_evict_config") = AdmitAndEvictConfig(),
             py::arg("initializer_random_pool_size") = -1, py::arg("seed") = 0, py::arg("num_features") = 1,
             py::arg("is_incremental") = false)
        .def_readwrite("table_name", &EmbConfig::tableName)
        .def_readwrite("initializer_type", &EmbConfig::initializerType)
        .def_readwrite("emb_dim", &EmbConfig::embDim)
        .def_readwrite("optim_num", &EmbConfig::optimNum)
        .def_readwrite("cache_size", &EmbConfig::cacheSize)
        .def_readwrite("weight_init_min", &EmbConfig::weightInitMin)
        .def_readwrite("weight_init_max", &EmbConfig::weightInitMax)
        .def_readwrite("weight_init_mean", &EmbConfig::weightInitMean)
        .def_readwrite("weight_init_stddev", &EmbConfig::weightInitStddev)
        .def_readwrite("admit_and_evict_config", &EmbConfig::admitAndEvictConfig)
        .def_readwrite("is_incremental", &EmbConfig::isIncremental);
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(embcache_pybind, m)
{
    AddInitializerType(m);

    py::class_<AdmitAndEvictConfig>(m, "AdmitAndEvictConfig")
        .def(py::init<>())
        .def(py::init<int64_t, float, uint64_t, uint64_t>(), py::arg("admit_threshold") = INVALID_KEY,
             py::arg("not_admitted_default_value") = 0.0, py::arg("evict_threshold") = 0,
             py::arg("evict_step_interval") = 0)
        .def_readwrite("admit_threshold", &AdmitAndEvictConfig::admitThreshold)
        .def_readwrite("not_admitted_default_value", &AdmitAndEvictConfig::notAdmittedDefaultValue)
        .def_readwrite("evict_threshold", &AdmitAndEvictConfig::evictThreshold)
        .def_readwrite("evict_step_interval", &AdmitAndEvictConfig::evictStepInterval);

    AddEmbConfigModule(m);

    py::class_<SwapInfo>(m, "SwapInfo")
        .def(py::init<>())  // 默认构造函数
        .def_readwrite("swapout_keys", &SwapInfo::swapoutKeys)
        .def_readwrite("swapout_offs", &SwapInfo::swapoutOffs)
        .def_readwrite("swapin_keys", &SwapInfo::swapinKeys)
        .def_readwrite("swapin_offs", &SwapInfo::swapinOffs)
        .def_readwrite("batch_offs", &SwapInfo::batchOffs)
        .def("get_swapin_keys_length_pre_sum", &SwapInfo::GetSwapinKeysLengthPreSum)
        .def("get_swapout_keys_length_pre_sum", &SwapInfo::GetSwapoutKeysLengthPreSum);

    py::class_<SwapinTensor>(m, "SwapinTensor")
        .def(py::init<>())  // 默认构造函数
        .def_readwrite("swapin_embs", &SwapinTensor::swapinEmbs)
        .def_readwrite("swapin_optims", &SwapinTensor::swapinOptims)
        .def_readwrite("jagged_offs", &SwapinTensor::jaggedOffs);

    AddEmbCacheManager(m);

    py::class_<AsyncTask<SwapInfo>>(m, "AsyncSwapInfo").def("get", &AsyncTask<SwapInfo>::get);
    py::class_<AsyncTask<SwapinTensor>>(m, "AsyncSwapinTensor").def("get", &AsyncTask<SwapinTensor>::get);
    py::class_<AsyncTask<void>>(m, "AsyncUpdate").def("get", &AsyncTask<void>::get);
    m.def("restore", &Restore, py::arg("unique_indices"), py::arg("unique_inverse"), py::arg("unique_offset"),
          py::arg("offsets"), py::arg("hash_indices"));
    m.def("restore_async", &RestoreAsync, py::arg("unique_indices"), py::arg("unique_inverse"),
          py::arg("unique_offset"), py::arg("offsets"), py::arg("hash_indices"));
}
