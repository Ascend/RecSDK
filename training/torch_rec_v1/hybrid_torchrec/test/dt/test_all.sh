#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

export GLOO_SOCKET_IFNAME="lo"
pytest ./ --cov=hybrid_torchrec --cov-branch --cov-report=term-missing --junitxml=final.xml --cov-report=xml:coverage.xml