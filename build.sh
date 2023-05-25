#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
# Description: build entrance script.

set -e
ROOT_DIR=$(dirname "$(readlink -f "$0")")

remove()
{
  if [ -d "$1" ]; then
    rm -rf "$1"
  elif [ -f "$1" ]; then
    rm -f "$1"
  fi
}

clean()
{
  remove "${ROOT_DIR}"/dist
  remove "${ROOT_DIR}"/install
  remove "${ROOT_DIR}"/mx_rec.egg-info
  remove "${ROOT_DIR}"/src/build
  remove "${ROOT_DIR}"/build/bdist.linux-"$(arch)"
  remove "${ROOT_DIR}"/build/tf1_env
  remove "${ROOT_DIR}"/build/tf2_env
  remove "${ROOT_DIR}"/build/lib
  remove "${ROOT_DIR}"/build/mindxsdk-mxrec
}

clean
