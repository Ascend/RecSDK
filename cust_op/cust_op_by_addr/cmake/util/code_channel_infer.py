#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""
import os
import stat
import ctypes
import collections
import shutil
import subprocess
import copy

"""CODE_* is used to cube/vector api is called in operator code
CODE_MIX means both cube and vector api is called
CODE_CUBE means only cube api is called
CODE_VEC means only vector api is called
"""
CODE_MIX = 0
CODE_CUBE = 1
CODE_VEC = 2


def _is_v220(op_product: str):
    """return if current soc version is V220

    Returns:
        res: True means V220
    """
    if op_product in ["ascend910b", "ascend910c"]:
        return True
    return False

CheckCoreTypeParams = collections.namedtuple('CheckCoreTypeParams',\
['src_file', 'arch', 'kernel_name', 'compile_options', 'addrspace_list', 'outdir'])


def _check_core_type(check_core_type_params: CheckCoreTypeParams):
    """1. call ccec -S -emit-llvm to generate llvm-ir file
       2. analysis addrspace to check if exists cube or vector buffer scope

    Args:
        CheckCoreTypeParams:
        src_file (str): TIK2 operator code file
        arch (str): _description_
        kernel_name (str): kernel function name
        compile_options (list): compile options for ccec cmd
        addrspace_list (list): addrspace of cube or vector
        outdir(str): temp file output

    Returns:
        res (bool): True if exists target addrspapce of arch
    """
    llvm_ir_file = os.path.join(check_core_type_params.outdir, check_core_type_params.kernel_name + "_" +\
        check_core_type_params.arch + ".ll")
    compile_cmd = [shutil.which("ccec"), '-S', '-emit-llvm', '-std=c++17', '-x', 'cce',\
        check_core_type_params.src_file]
    compile_cmd += check_core_type_params.compile_options

    compile_cmd += ["--cce-aicore-arch=%s" % check_core_type_params.arch,
                    "--cce-aicore-only", "-o", llvm_ir_file]
    proc = subprocess.Popen(
        compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "compile %s error :%s\n" % (check_core_type_params.src_file, out.decode())
        print("check core type ", msg)
        return False

    def _check_exist_space(line_content, space_list):
        if "addrspace" not in line_content:
            return False
        for space in space_list:
            if space in line_content:
                return True
        return False

    access_space = False
    with open(llvm_ir_file) as llvm_ir:
        line_list = llvm_ir.readlines()
        for line in line_list:
            if access_space:
                break
            access_space = _check_exist_space(line, check_core_type_params.addrspace_list)
    os.remove(llvm_ir_file)
    return access_space


InfoCodeChanelParams = collections.namedtuple('InfoCodeChanelParams',\
['src_file', 'tiling_header', 'kernel_name', 'outdir', 'op_product', 'compile_options'])


def infer_code_channel(params: InfoCodeChanelParams):
    """get code channel for v220, return CODE_MIX if soc version is not V220

    Args:
        src_file (str): TIK2 operator code file
        src_file (str): TIK2 operator tiling header file
        kernel_name (str): kernel function name
        optype (str): operator type
        compile_options (list): compile options for ccec cmd

    Raises:
        Exception: if not exist L1/L0/UB if code, it's not a aicore code

    Returns:
        res (int): CODE_MIX/CODE_CUBE/CODE_VEC
    """
    if not _is_v220(params.op_product):
        return CODE_MIX
    if params.compile_options is None:
        compile_options = []
    else:
        compile_options = params.compile_options
    ccec = shutil.which("ccec")
    if ccec is not None:
        ccec_path = os.path.dirname(ccec)
        tikcpp_path = os.path.realpath(os.path.join(ccec_path, "..", "..", "tikcpp"))
    else:
        tikcpp_path = os.path.realpath("/usr/local/Ascend/latest/compiler/tikcpp")
    compile_options.append("-I" + tikcpp_path)
    compile_options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    compile_options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    compile_options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    compile_options.append("-D__NPU_TILING__")
    compile_options += ["-include", params.tiling_header]
    cube_addrspace_list = ["addrspace(2)", "addrspace(3)", "addrspace(4)", "addrspace(5)"]
    access_l1_l0a = _check_core_type(CheckCoreTypeParams(params.src_file, "dav-c220-cube", params.kernel_name,\
        compile_options, cube_addrspace_list, params.outdir))
    cube_addrspace_list = ["addrspace(6)"]
    access_ub = _check_core_type(CheckCoreTypeParams(params.src_file, "dav-c220-vec", params.kernel_name,\
        compile_options, cube_addrspace_list, params.outdir))

    if access_l1_l0a and access_ub:
        return CODE_MIX
    elif access_l1_l0a:
        return CODE_CUBE
    elif access_ub:
        return CODE_VEC
    else:
        raise Exception(f"cannot find valid addrspace in (2,3,4,5,6) in {params.src_file}")
