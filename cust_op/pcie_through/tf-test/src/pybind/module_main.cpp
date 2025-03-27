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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "rma_swap/rma_swap_test.h"
#include "host_mgmt/host_mgmt.h"
#include "tdt_transfer/tdt_transfer.h"

namespace py = pybind11;

namespace {
    void GetShmInfo(py::module_& m);

    void GetHostMgmt(py::module_& m);

    PYBIND11_MODULE(custom_pybind, m)
{
    m.def("get_shm_mem", &MxRec::GetShmAddr, py::arg("name"), py::arg("deviceId"), py::arg("capacity"));

    m.def("read_shm_mem", &TfTest::ReadShmData, py::arg("name"), py::arg("save") = false, py::arg("savePath") = "./");

    m.def("write_shm_mem", &TfTest::WriteShmData, py::arg("name"), py::arg("shape"),
    py::arg("save") = false, py::arg("savePath") = "./");

    m.def("free_shm_mem", &MxRec::FreeShmAddr, py::arg("deviceId"));

    GetShmInfo(m);

    GetHostMgmt(m);
}


void GetShmInfo(py::module_& m)
{
    pybind11::class_<TfTest::ShmInfo>(m, "ShmInfo")
        .def(py::init<std::string, uint64_t, int, int, std::vector<int64_t>>(), py::arg("channel_name"),
             py::arg("mem_size"), py::arg("capacity"), py::arg("device_id"), py::arg("shape"))
        .def_readwrite("channel_name", &TfTest::ShmInfo::channelName)
        .def_readwrite("mem_size", &TfTest::ShmInfo::memSize)
        .def_readwrite("capacity", &TfTest::ShmInfo::capacity)
        .def_readwrite("device_id", &TfTest::ShmInfo::deviceId)
        .def_readwrite("shape", &TfTest::ShmInfo::shape);
}

void GetHostMgmt(py::module_& m)
{
    pybind11::class_<TfTest::HostMgmt>(m, "HostMgmt")
        .def(py::init())
        .def("initialize", &TfTest::HostMgmt::Initialize, py::arg("shm_infos"), py::arg("step_num"),
             py::arg("seed"), py::arg("type") = "RMA")
        .def("destroy", &TfTest::HostMgmt::Destroy);
}

}