#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

"""torchrec 版本检测逻辑。

支持多种版本获取方式，按优先级依次尝试：
1. torchrec.__version__
2. torchrec.version.__version__
3. torchrec 包目录下的 version.txt
4. importlib.metadata.version("torchrec")
"""

from importlib import metadata
from pathlib import Path
from typing import Tuple

import torchrec


def _torchrec_version() -> str:
    version = getattr(torchrec, "__version__", None)
    if version is not None:
        return str(version)
    try:
        from torchrec.version import __version__ as version

        if version is not None:
            return str(version)
    except Exception:
        pass  # pass Exception, try other method  # nosec B110

    try:
        for package_path in getattr(torchrec, "__path__", []):
            version_path = Path(package_path) / "version.txt"
            if version_path.exists():
                return version_path.read_text().strip()
    except Exception:
        pass  # pass Exception, try other method  # nosec B110

    try:
        return metadata.version("torchrec")
    except Exception:
        raise RuntimeError("Can not get torchrec version")


def _torchrec_version_tuple() -> Tuple[int, int, int]:
    version = _torchrec_version().split("+", 1)[0]
    parts = []
    for item in version.split(".")[:3]:
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


TORCH_REC_VERSION = _torchrec_version_tuple()
