#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co.,Ltd. 2025. All rights reserved.

from pathlib import Path

_MAX_PATH_LENGTH = 1024
_DEFAULT_BLACK_DIRS = ["/usr/bin", "/usr/bin", "/usr/sbin", "/etc", "/usr/lib", "/usr/lib64", "/usr/local"]
_DEFAULT_SENSITIVE_WORDS = ["key", "password", "privatekey"]


def check_input_path_valid(path: str):
    if not path or not isinstance(path, str):
        raise TypeError(f"Input path {path} is not valid str")
    if len(path) > _MAX_PATH_LENGTH:
        raise ValueError(f"Input path {path[: _MAX_PATH_LENGTH]} length over limit")
    if ".." in path:
        raise ValueError(f"theres are illegal characters '..' in path: {path} ")
    # black dirs check
    is_start_with_black_dirs = any([path.startswith(item) for item in _DEFAULT_BLACK_DIRS])
    if is_start_with_black_dirs:
        raise ValueError(f"path can't start with black dirs, but got:{path}")
    # sensitive word check
    contains_sensitive_word = any([item in path.lower() for item in _DEFAULT_SENSITIVE_WORDS])
    if contains_sensitive_word:
        raise ValueError(f"path can't contains sensitive words, but got:{path}")
    # link file check
    if Path(path).resolve() != Path(path).absolute():
        raise ValueError(f"soft link path can't be a path param, but got:{path}")
