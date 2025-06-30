#!/bin/bash
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

curr_path=$(cd $(dirname $0); pwd)
cd ${curr_path}

function check_ret_fn()
{
    if [ $? -ne 0 ]; then
        echo "[FAIL] $@ failed"
        exit -1
    else
        echo "[SUCCESS] $@ successful"
    fi 
}

function check_whl_pkg_exist_fn() 
{
  whl_pkg="./dist/torchrec_embcache-*.whl"
  if [ ! -f ${whl_pkg} ]; then
    echo "[ERROR] ${whl_pkg} does not exist, please check!"
    exit -1
  else
    echo "[SUCCESS] ${whl_pkg} exists!"
  fi
}
 
function un_in_stall_fn()
{
  cd ./dist
  pip list|grep torchrec-embcache
  if [ $? -ne 0 ]; then
    echo "torchrec-embcache is not installed, no need to uninstall"
  else
      pip3 uninstall -y torchrec-embcache
      check_ret_fn "uninstall"
  fi
  
  pip3 install ./torchrec_embcache-*.whl --no-deps --force-reinstall
  check_ret_fn "install"
}

check_whl_pkg_exist_fn
un_in_stall_fn