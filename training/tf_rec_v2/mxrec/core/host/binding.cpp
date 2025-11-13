/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "runtime/runtime_manager.h"

namespace py = pybind11;

namespace {

PYBIND11_MODULE(host, m)
{
    m.doc() = "Python bindings for host runtime manager.";

    py::class_<rec_sdk::runtime::RuntimeManager>(m, "RuntimeManager")
        .def(py::init<int32_t>())
        .def("start_count_filter", &rec_sdk::runtime::RuntimeManager::StartCountFilter, "Start count filter.",
             py::arg("table_name"), py::arg("min_used_times"))
        .def("start_time_evictor", &rec_sdk::runtime::RuntimeManager::StartTimeEvictor, "Start time evictor.",
             py::arg("table_name"), py::arg("max_cold_secs"))
        .def("save_count_filter", &rec_sdk::runtime::RuntimeManager::SaveCountFilter, "Save count filter.",
             py::arg("table_name"), py::arg("file_path"))
        .def("save_time_evictor", &rec_sdk::runtime::RuntimeManager::SaveTimeEvictor, "Save time evictor.",
             py::arg("table_name"), py::arg("file_path"))
        .def("load_count_filter", &rec_sdk::runtime::RuntimeManager::LoadCountFilter, "Load count filter.",
             py::arg("table_name"), py::arg("file_path"))
        .def("load_time_evictor", &rec_sdk::runtime::RuntimeManager::LoadTimeEvictor, "Load time evictor.",
             py::arg("table_name"), py::arg("file_path"))
        .def("get_evicted_keys", &rec_sdk::runtime::RuntimeManager::GetEvictedKeys, "Get evicted keys.",
             py::arg("table_name"));
}

}  // namespace
