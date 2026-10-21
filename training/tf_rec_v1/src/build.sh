#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e
[ -d build ] && rm -rf build;
mkdir build && cd build || exit 1
# HDF5_PATH is optional
python_bin="$(which python3.7)"
python_path="$(dirname "$(dirname "$(realpath "$python_bin")")")"
if [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    ascend_path=/usr/local/Ascend/ascend-toolkit/latest
elif [ -d /usr/local/Ascend/latest ]; then
    ascend_path=/usr/local/Ascend/latest
else
    echo "ERROR: can not find toolkit and tfplugin"
    exit 1
fi

# locate openmpi prefix: prefer OpenMPI's own query, then mpicc-derived path,
# then common install dirs (source build /usr/local/openmpi, yum /usr/lib64/openmpi)
ompi_path=""
if command -v mpirun >/dev/null 2>&1; then
    ompi_path="$(mpirun --showme:prefix 2>/dev/null || true)"
fi
if [ -z "$ompi_path" ] && command -v mpicc >/dev/null 2>&1; then
    ompi_path="$(dirname "$(dirname "$(readlink -f "$(command -v mpicc)")")")"
fi
if [ -n "$ompi_path" ] && [ ! -f "$ompi_path/include/mpi.h" ] && [ ! -x "$ompi_path/bin/mpicc" ]; then
    ompi_path=""
fi
if [ -z "$ompi_path" ]; then
    for candidate in /usr/local/openmpi /usr/lib64/openmpi; do
        if [ -f "$candidate/include/mpi.h" ] || [ -x "$candidate/bin/mpicc" ]; then
            ompi_path="$candidate"
            break
        fi
    done
fi
if [ -z "$ompi_path" ]; then
    echo "ERROR: openmpi not found"
    exit 1
fi

# pybind11 查找顺序：已安装(import 直接可用) → opensource 源码树 → pip 安装
pybind_args=""
opensource_dir="$2"/../opensource
if "$python_bin" -c "import pybind11" 2>/dev/null; then
    # 1. 已安装：走 find_package（传入 pybind11_DIR / CMAKE_PREFIX_PATH + Python3 锁定）
    pybind_cmake_dir=$("$python_bin" -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)
    if [ -n "$pybind_cmake_dir" ]; then
        echo "[PYBIND11-FALLBACK] pybind11 already importable, use installed cmake dir: $pybind_cmake_dir"
        pybind_args="-Dpybind11_DIR=$pybind_cmake_dir -DCMAKE_PREFIX_PATH=$pybind_cmake_dir -DPYBIND11_FINDPYTHON=ON -DPython3_EXECUTABLE=$python_bin -DPython3_ROOT_DIR=$python_path"
    else
        echo "[PYBIND11-FALLBACK] ERROR: import pybind11 ok but cmake dir not found"
        exit 1
    fi
elif [ -d "${opensource_dir}/pybind11" ]; then
    # 2. opensource 源码树：交给 CMakeLists 的 add_subdirectory
    echo "[PYBIND11-FALLBACK] opensource pybind11 source tree exists, use add_subdirectory"
else
    # 3. pip 安装后走 find_package
    echo "[PYBIND11-FALLBACK] pybind11 not importable and no opensource, installing via pip..."
    "$python_bin" -m pip install "pybind11>=2.10.3"
    pybind_cmake_dir=$("$python_bin" -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)
    if [ -n "$pybind_cmake_dir" ]; then
        echo "[PYBIND11-FALLBACK] pip-installed pybind11 cmake dir: $pybind_cmake_dir"
        pybind_args="-Dpybind11_DIR=$pybind_cmake_dir -DCMAKE_PREFIX_PATH=$pybind_cmake_dir -DPYBIND11_FINDPYTHON=ON -DPython3_EXECUTABLE=$python_bin -DPython3_ROOT_DIR=$python_path"
    else
        echo "[PYBIND11-FALLBACK] ERROR: pybind11 cmake dir not found"
        exit 1
    fi
fi

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DTF_PATH="$1" \
    -DOMPI_PATH="$ompi_path" \
    -DPYTHON_PATH="$python_path" \
    -DASCEND_PATH="$ascend_path" \
    -DABSEIL_PATH="$1" \
    -DSECUREC_PATH="$2"/../opensource/securec \
    -DCMAKE_INSTALL_PREFIX="$2"/tf_rec_v1_output \
    -DBUILD_CUST="$3" \
    ${pybind_args} ..
make -j8
make install
