#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import List

_ABS_PATH_MIN_LEN = 1
_ABS_PATH_MAX_LEN = 1024
_FILE_NAME_MAX_LEN = 200  # max file name size is 255 bytes, reserve some bytes

_DEFAULT_BLACK_DIRS = ["/usr/bin", "/usr/bin", "/usr/sbin", "/etc", "/usr/lib", "/usr/lib64", "/usr/local"]
_DEFAULT_SENSITIVE_WORDS = ["Key", "password", "privatekey"]


def check_str_type(string_value: str) -> None:
    if not isinstance(string_value, str):
        raise TypeError(f"expected param type string but got {type(string_value)}")


def check_str_type_and_len(string_value: str, min_length: int, max_length: int) -> None:
    check_str_type(string_value)
    if len(string_value) < min_length or len(string_value) > max_length:
        raise ValueError(f"string param length is invalid, got param length:{len(string_value)},"
                         f" length limit:[{min_length}, {max_length}]")


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

    check_str_type(value)
    value = os.path.realpath(value)
    if os.path.abspath(value) != os.path.realpath(value):
        raise ValueError(f"soft link or relative path can't be a path param, got:{value}")
    if not Path(value).is_absolute():
        check_str_type_and_len(value, 0, _FILE_NAME_MAX_LEN)
    check_str_type_and_len(os.path.abspath(value), _ABS_PATH_MIN_LEN, _ABS_PATH_MAX_LEN)
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

    _check_path_permission(value)
    file_exist = os.path.exists(os.path.realpath(value))
    current_is_dir = file_exist and os.path.isdir(value)
    if is_dir and not current_is_dir:
        raise ValueError(f"expected path param is a directory, but file not exist or not a directory")

    if file_exist and not os.path.isdir(value):
        file_bytes = Path(value).stat().st_size
        if file_size_min and file_bytes < file_size_min:
            raise ValueError(f"file size:{file_bytes} in byte is slower than file min size:{file_size_min}")
        if file_size_max and file_bytes > file_size_max:
            raise ValueError(f"file size::{file_bytes} in byte exceeds file max size limit:{file_size_max}")


def _check_path_permission(file_path: str):
    realpath = os.path.realpath(file_path)
    path = Path(realpath)
    last_exist_parent = ""
    for ancestor in [*path.parents]:
        if ancestor.exists():
            last_exist_parent = ancestor.absolute()
            break
    if not last_exist_parent:
        raise ValueError(f"check path permission error, there is not exist at least one parent path for: {realpath}")

    # 检查权限
    process_uid = os.geteuid()
    process_gid = os.getegid()
    stat_info = os.stat(last_exist_parent)
    file_uid = stat_info.st_uid
    file_gid = stat_info.st_gid
    if not (process_uid == file_uid or process_gid == file_gid):
        raise ValueError(f"current user don't have access permission for the path:{last_exist_parent}")
