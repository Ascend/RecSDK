#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import shutil
import stat
import os
import subprocess
import sys
import logging
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


logging.basicConfig(level=logging.INFO)


def modify_version():
    default_version = "7.3+t50"

    init_file = "dynamic_emb/__init__.py"
    with open(init_file, "r") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "__version__ = " not in line:
                continue
            lines[idx] = f"__version__ = '{default_version}'\n"
            break

    flag = os.O_WRONLY | os.O_TRUNC
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(init_file, flag, mode), "w") as out:
        out.writelines(lines)
    return default_version


def ensure_pybind11():
    """确保pybind11可用"""
    try:
        import pybind11
        return pybind11.get_cmake_dir()
    except ImportError:
        logging.info("Installing pybind11...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
        import pybind11
        return pybind11.get_cmake_dir()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ('pybind11-dir=', None, 'Path to pybind11 installation'),
    ]
    
    def initialize_options(self):
        super().initialize_options()
        self.pybind11_dir = None

    def finalize_options(self):
        super().finalize_options()
    
    def run(self):
        # 确保pybind11可用
        if not self.pybind11_dir:
            self.pybind11_dir = ensure_pybind11()

        # 首先确保CMake可用
        try:
            cmake_exe = shutil.which("cmake")
            if cmake_exe is None:
                raise RuntimeError("CMake must be installed to build the extensions")
            subprocess.check_output([cmake_exe, '--version'])
        except OSError as e:
            raise RuntimeError("CMake must be installed to build the extensions") from e
        
        super().run()
        self.copy_cust_ops_libraries()
        
    def copy_cust_ops_libraries(self):
        """复制自定义算子库到包目录"""
        ext_path = self.get_ext_fullpath('dynamic_emb_extensions')
        package_dir = os.path.dirname(ext_path)
        
        # 查找依赖库
        lib_patterns = [
            'libdynamic_emb_op_*.so',
            'libasc_kernel_lib.so'
        ]
        
        # 搜索路径
        search_paths = [
            os.path.join(self.build_temp, 'lib'),
        ]
        
        for pattern in lib_patterns:
            for search_path in search_paths:
                if os.path.exists(search_path):
                    for lib_file in Path(search_path).glob(pattern):
                        if lib_file.is_file():
                            lib_dest_path = os.path.join(package_dir, lib_file.name)
                            logging.info(f"Copying {lib_file} to {lib_dest_path}")
                            shutil.copy2(lib_file, lib_dest_path)
    
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)
    
    def build_cmake_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake配置参数
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
            f'-Dpybind11_DIR={self.pybind11_dir}',
            f'-DCMAKE_PREFIX_PATH={self.pybind11_dir}'
        ]
        
        # 从环境变量获取配置或使用默认值
        run_mode = os.getenv('RUN_MODE', 'npu')
        soc_version = os.getenv('SOC_VERSION', 'Ascend910_9579')
        ascend_cann_path = os.getenv('ASCEND_CANN_PACKAGE_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
        
        cmake_args.extend([
            f'-DRUN_MODE={run_mode}',
            f'-DSOC_VERSION={soc_version}',
            f'-DASCEND_CANN_PACKAGE_PATH={ascend_cann_path}'
        ])
        
        # 构建目录
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        
        logging.info("Configuring CMake project...")
        logging.info(f"CMake args: {cmake_args}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        
        logging.info("Building project...")
        subprocess.check_call(['cmake', '--build', '.', '--verbose', '--config', 'Debug' if self.debug else 'Release'], 
                             cwd=build_temp)

# 设置依赖
setup_requires = ['pybind11']


class CustomInstall(install):
    def run(self):
        super().run()
        # 设置库文件权限
        self.fix_library_permissions()
    
    def fix_library_permissions(self):
        """修复库文件权限"""
        package_dir = self.install_lib
        if package_dir and os.path.exists(package_dir):
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    if file.endswith('.so'):
                        file_path = os.path.join(root, file)
                        os.chmod(file_path, 0o755)  # 设置可执行权限

 # 编译 so文件
script_path = os.getcwd()
common_script = os.path.join(script_path, "./scripts/build.sh")
os.chmod(common_script, 0o755)
res = subprocess.run([common_script], shell=False)
if res.returncode:
    raise RuntimeError("compile so files failed!")

# 安装common
common_dir = os.path.join(script_path, "../../common")
subprocess.run(
    [
        "python3",
        "setup.py",
        "bdist_wheel",
    ],
    cwd=common_dir,
    shell=False,
)
if os.path.exists("rec_sdk_common"):
    shutil.rmtree("rec_sdk_common")
current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(current_dir, "..", "..", "common", "rec_sdk_common")
dest_path = os.path.join(current_dir, "rec_sdk_common")

if os.path.exists(source_path):
    shutil.copytree(source_path, dest_path)

# 将本地包名替换为目标包名
TARGET_PACKAGE_NAME = "dynamic_emb"
version = modify_version()
local_packages = find_packages(exclude=("tests*", "*test"))

target_packages = []
package_dir_mapping = {}
for local_pkg in local_packages:
    target_pkg = local_pkg.replace("dynamic_emb", TARGET_PACKAGE_NAME, 1)
    target_packages.append(target_pkg)
    package_dir_mapping[target_pkg] = local_pkg.replace(".", os.sep)

setup(
    name="dynamic_emb",
    version=version,
    packages=target_packages,
    package_dir=package_dir_mapping,
    ext_modules=[CMakeExtension("dynamic_emb_extensions", "./")],
    cmdclass={
        'build_ext': CMakeBuild,
        'install': CustomInstall
    },
    package_data={
        TARGET_PACKAGE_NAME: ['*.so'],  # 主包下的.so
        "rec_sdk_common": ['lib/*.so'],
    },
    include_package_data=True,
    setup_requires=setup_requires,
    install_requires=['pybind11'],
    zip_safe=False,
)

if 'bdist_wheel' in sys.argv:
    move_whl_script = os.path.join(
        script_path, "./scripts/move_whl_file_2_pkg_dir.sh"
    )
    os.chmod(move_whl_script, 0o755)
    res = subprocess.run([move_whl_script], shell=False)
    if res.returncode:
        raise RuntimeError("move whl file to pkg dir failed!")

    gen_tar_script = os.path.join(
        script_path, "./scripts/gen_tar_pkg.sh"
    )
    os.chmod(gen_tar_script, 0o755)
    res = subprocess.run([gen_tar_script], shell=False)
    if res.returncode:
        raise RuntimeError("gen dynamicemb's tar pkg failed!")
