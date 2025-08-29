#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import List

_STRING_MIN_LEN = 0
_STRING_MAX_LEN = 1024

_DEFAULT_BLACK_DIRS = ["/usr/bin", "/usr/bin", "/usr/sbin", "/etc", "/usr/lib", "/usr/lib64", "/usr/local"]
_DEFAULT_SENSITIVE_WORDS = ["Key", "password", "privatekey"]


def check_str(string_value: str, min_length: int, max_length: int) -> None:
    if not isinstance(string_value, str):
        raise TypeError(f"expected param type string but got {type(string_value)}")
    if len(string_value) < min_length or len(string_value) > max_length:
        raise ValueError(f"string param length is invalid, length limit:[{min_length}, {max_length}]")


def check_path(value: str, need_exist: bool = False, is_dir: bool = False, **kwargs) -> None:
    """
    Check path whether valid.

    Args:
        value (str): check path str.
        need_exist (bool): check path need exist if True, default is False.
        is_dir (bool): Check that the path needs to be a directory if True, default is False.
        **kwargs: other parameter dict.
            file_size_min: int = 0,
            file_size_max: int = 0,
            black_dirs: List[str] = None,
            sensitive_words: List[str] = None

    Returns:
        None.
    """
    file_size_min: int = kwargs.get("file_size_min", 0)
    file_size_max: int = kwargs.get("file_size_max", 0)
    black_dirs: List[str] = kwargs.get("black_dirs", [])
    sensitive_words: List[str] = kwargs.get("sensitive_words", [])

    check_str(value, _STRING_MIN_LEN, _STRING_MAX_LEN)
    if os.path.abspath(value) != os.path.realpath(value):
        raise ValueError(f"soft link or relative path can't be a path param, got:{value}")
    if need_exist and not os.path.exists(os.path.realpath(value)):
        raise ValueError(f"expected path exist, but got:{value}")

    black_dirs = black_dirs or _DEFAULT_BLACK_DIRS
    is_start_with_black_dirs = any([value.startswith(item) for item in black_dirs])
    if is_start_with_black_dirs:
        raise ValueError(f"path can't start with black dirs, but got:{value}")

    sensitive_words = sensitive_words or _DEFAULT_SENSITIVE_WORDS
    contains_sensitive_word = any([item in value for item in sensitive_words])
    if contains_sensitive_word:
        raise ValueError(f"path can't contains sensitive words, but got:{value}")

    file_exist = os.path.exists(os.path.realpath(value))
    # 检查权限
    if file_exist:
        process_uid = os.geteuid()
        process_gid = os.getegid()
        stat_info = os.stat(value)
        file_uid = stat_info.st_uid
        file_gid = stat_info.st_gid
        if not (process_uid == file_uid or process_gid == file_gid):
            raise ValueError(f"current user don't have access permission for the path:{value}")

    current_is_dir = file_exist and os.path.isdir(value)
    if is_dir and not current_is_dir:
        raise ValueError(f"expected path param is a directory, but file not exist or not a directory")

    file_exist = os.path.exists(os.path.realpath(value))
    if file_exist and not os.path.isdir(value):
        file_bytes = Path(value).stat().st_size
        if file_size_min and file_bytes < file_size_min:
            raise ValueError(f"file size:{file_bytes} in byte is slower than file min size:{file_size_min}")
        if file_size_max and file_bytes > file_size_max:
            raise ValueError(f"file size::{file_bytes} in byte exceeds file max size limit:{file_size_max}")
