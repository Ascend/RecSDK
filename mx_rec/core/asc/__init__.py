#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

__all__ = ["get_asc_insert_func", "start_asc_pipeline", "FeatureSpec"]

from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.asc.helper import get_asc_insert_func
