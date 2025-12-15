#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess
from pathlib import Path
from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        sha = subprocess.check_output(cmd, cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
            version = f.readline().strip()

    if sha is None:
        sha = "Unknown"
    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "src" / "torchrec_embcache" / "version.py"
    with open(version_path, "w") as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))
    

def main() -> None:
    version, sha = _get_version()
    _export_version(version, sha)

    setup(
        name="torchrec_embcache",
        version=version,
        author="MindX Team",
        author_email="",
        description="Pytorch embedding cache implementation for recommendation systems",
        url="",
        license="BSD-3",
        keywords=["pytorch", "embedding cache"],
        package_data={
            'torchrec_embcache': ['*.so*']
        },
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.10",
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
