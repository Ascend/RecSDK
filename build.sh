#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
# Description: build entrance script.

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
bash "${SCRIPT_DIR}"/build/build.sh
