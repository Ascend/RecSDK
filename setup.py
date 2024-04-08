#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import shutil
import subprocess

# clean pkg_dir existed
PKG_DIR = "./build/mindxsdk-mxrec"
if os.path.exists(PKG_DIR):
    shutil.rmtree(PKG_DIR)

# build tf1's wheel file
res = subprocess.run(["python3.7", "setup_tf1.py", "bdist_wheel"], shell=False)
if res.returncode:
    raise RuntimeError(f"build tf1's wheel file failed!")

# build tf2's wheel file
res = subprocess.run(["python3.7", "setup_tf2.py", "bdist_wheel"], shell=False)
if res.returncode:
    raise RuntimeError(f"build tf2's wheel file failed!")

# copy cust_op, examples files, etc. Then gen mxrec's tar pkg
res = subprocess.run(["bash", "./build/gen_mxrec_tar_pkg.sh"], shell=False)
if res.returncode:
    raise RuntimeError(f"gen mxrec's tar pkg failed!")
