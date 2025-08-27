#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

from rec_sdk_common.constants.constants import ValidatorParams
from mx_rec.validator.validator import FileValidator

def file_safe_check(
        name: str,
        path: str,
        unsupported_mode: int = 0o022,
        min_size: int = ValidatorParams.FILE_MIN_SIZE.value,
        max_size: int = ValidatorParams.FILE_MAX_SIZE.value,
):
    validator = FileValidator(name, path)
    validator.check_not_soft_link().check_file_size(max_size, min_size).check_file_mode(
        unsupported_mode
    ).check_user_group().check()