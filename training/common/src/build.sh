#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
python_path="$(dirname "$(dirname "$(realpath "$(which python3.7)")")")"
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

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPYTHON_PATH="$python_path" \
    -DASCEND_PATH="$ascend_path" \
    -DOMPI_PATH="$ompi_path" \
    -DSECUREC_PATH="$1"/../opensource/securec \
    -DCMAKE_INSTALL_PREFIX="$1"/common_output \
    -DBUILD_CUST="$2" ..
make -j8
make install
