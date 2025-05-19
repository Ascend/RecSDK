#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List
import logging

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
    version_path = ROOT_DIR / "hybrid_torchrec" / "version.py"
    with open(version_path, "w") as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="hybrid_torchrec setup")
    try:
        args, unknown = parser.parse_known_args(argv)
        return args, unknown
    except (argparse.ArgumentError, SystemExit) as e:
        # 处理 argparse.ArgumentError 和 SystemExit 异常
        print(f"Error: {e}")
        return None, None


def main(argv: List[str]) -> None:
    _, unknown = parse_args(argv)
    if unknown is None:
        # 如果 parse_args 返回 None，表示解析参数时发生错误
        sys.exit(1)

    with open(os.path.join(ROOT_DIR, "README.MD"), encoding="utf8") as f:
        readme = f.read()
    with open(os.path.join(ROOT_DIR, "install-requirements.txt"), encoding="utf8",) as f:
        reqs = f.read()
        install_requires = reqs.strip().split("\n")

    version, sha = _get_version()
    _export_version(version, sha)

    packages = find_packages(
        exclude=(
            "*tests",
            "*test",
            "examples",
            "*examples.*",
            "*benchmarks",
            "*build",
            "*rfc",
        )
    )
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name="hybrid_torchrec",
        version=version,
        author="MindX Team",
        author_email="",
        description="Pytorch domain library for recommendation systems",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="",
        license="BSD-3",
        keywords=["pytorch", "recommendation systems", "sharding"],
        python_requires=">=3.8",
        install_requires=install_requires,
        packages=packages,
        package_data={'': ['*.so*']},
        zip_safe=False,
        # PyPI package information.
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )



if __name__ == "__main__":
    main(sys.argv[1:])
