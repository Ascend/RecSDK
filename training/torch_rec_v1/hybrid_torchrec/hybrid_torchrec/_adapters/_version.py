#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
