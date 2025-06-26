/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
#include <any>
#include <chrono>
#include <vector>

#include "../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/embedding_cache/embcache_manager.h"
#include "../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/ops/restore.h"

using namespace Embcache;

// Registers _C as a Python extension module.
PYBIND11_MODULE(embcache_pybind, m)
{
    py::class_<AdmitAndEvictConfig>(m, "AdmitAndEvictConfig")
        .def(py::init<>())
        .def(py::init<int32_t, float, uint64_t, uint64_t>(), py::arg("admit_threshold") = -1,
             py::arg("not_admitted_default_value") = 0.0, py::arg("evict_threshold") = 0,
             py::arg("evict_step_interval") = 0)
        .def_readwrite("admit_threshold", &AdmitAndEvictConfig::admitThreshold)
        .def_readwrite("not_admitted_default_value", &AdmitAndEvictConfig::notAdmittedDefaultValue)
        .def_readwrite("evict_threshold", &AdmitAndEvictConfig::evictThreshold)
        .def_readwrite("evict_step_interval", &AdmitAndEvictConfig::evictThreshold);

    py::class_<EmbConfig>(m, "EmbConfig")
        .def(py::init<>())
        .def(py::init<const std::string&, int32_t, int32_t, int64_t, float, float, AdmitAndEvictConfig>(),
             py::arg("table_name"), py::arg("emb_dim"), py::arg("optim_num"), py::arg("cache_size"),
             py::arg("weight_init_min"), py::arg("weight_init_max"),
             py::arg("admit_and_evict_config") = AdmitAndEvictConfig())
        .def_readwrite("table_name", &EmbConfig::tableName)
        .def_readwrite("emb_dim", &EmbConfig::embDim)
        .def_readwrite("optim_num", &EmbConfig::optimNum)
        .def_readwrite("cache_size", &EmbConfig::cacheSize)
        .def_readwrite("weight_init_min", &EmbConfig::weightInitMin)
        .def_readwrite("weight_init_max", &EmbConfig::weightInitMax)
        .def_readwrite("admit_and_evict_config", &EmbConfig::admitAndEvictConfig);

    py::class_<SwapInfo>(m, "SwapInfo")
        .def(py::init<>())  // 默认构造函数
        .def_readwrite("swapout_keys", &SwapInfo::swapoutKeys)
        .def_readwrite("swapout_offs", &SwapInfo::swapoutOffs)
        .def_readwrite("swapin_keys", &SwapInfo::swapinKeys)
        .def_readwrite("swapin_offs", &SwapInfo::swapinOffs)
        .def_readwrite("batch_offs", &SwapInfo::batchOffs);

    py::class_<SwapinTensor>(m, "SwapinTensor")
        .def(py::init<>())  // 默认构造函数
        .def_readwrite("swapin_embs", &SwapinTensor::swapinEmbs)
        .def_readwrite("swapin_optims", &SwapinTensor::swapinOptims)
        .def_readwrite("jagged_offs", &SwapinTensor::jaggedOffs);

    py::class_<EmbcacheManager>(m, "EmbcacheManager")
        .def(py::init<const std::vector<EmbConfig>&>(), py::arg("emb_configs"))

        .def("compute_swap_info_async", &EmbcacheManager::ComputeSwapInfoAsync, py::arg("batch_keys"),
             py::arg("jagged_offs"))

        .def("save", &EmbcacheManager::Save, py::arg("path"), py::arg("rank"))

        .def("embedding_to_host", &EmbcacheManager::Embedding2Host, py::arg("weights_dev"), py::arg("momentum1_dev"))

        .def("embedding_lookup_async", &EmbcacheManager::EmbeddingLookupAsync, py::arg("swap_info"))

        .def("embedding_update_async", &EmbcacheManager::EmbeddingUpdateAsync, py::arg("swap_info"),
             py::arg("swapout_embs"), py::arg("swapout_optims"))

        .def("load", &EmbcacheManager::Load, py::arg("path"), py::arg("rank"))

        .def("record_timestamp", &EmbcacheManager::RecordTimestamp, py::arg("batch_keys"), py::arg("jagged_offs"),
             py::arg("batch_timestamps"))
        .def("evict_features", &EmbcacheManager::EvictFeatures)
        .def("statistics_key_count", &EmbcacheManager::StatisticsKeyCount, py::arg("batch_keys"), py::arg("offset"),
             py::arg("batch_key_counts"), py::arg("table_index"));

    py::class_<AsyncTask<SwapInfo>>(m, "AsyncSwapInfo").def("get", &AsyncTask<SwapInfo>::get);
    py::class_<AsyncTask<SwapinTensor>>(m, "AsyncSwapinTensor").def("get", &AsyncTask<SwapinTensor>::get);
    py::class_<AsyncTask<void>>(m, "AsyncUpdate").def("get", &AsyncTask<void>::get);
    m.def("restore", &Restore, py::arg("unique_indices"), py::arg("unique_inverse"), py::arg("unique_offset"),
          py::arg("offsets"), py::arg("hash_indices"));
    m.def("restore_async", &RestoreAsync, py::arg("unique_indices"), py::arg("unique_inverse"),
          py::arg("unique_offset"), py::arg("offsets"), py::arg("hash_indices"));
}