# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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


import multiprocessing
import os
import re
import subprocess
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("args", None, "Arguments to pass to the script")


def run_script(base_device, total_devices, base_args):
    script_directory = os.path.join(os.path.dirname(__file__))

    main_script_path = os.path.join(script_directory, "main.py")

    cmd = ["python3", main_script_path, f"--device={base_device}", f"--total_devices={total_devices}"]
    cmd.extend(base_args)
    subprocess.run(cmd)


def parse_args(args_string):
    base_args = []
    total_devices = 1
    base_device = "npu"
    for arg_v in re.split(r"\s+--", args_string.strip()):
        arg_v = arg_v.strip("--")
        arg, v = re.split(r"\s+|=", arg_v)
        if arg == "base_device":
            base_device = v
        elif arg == "total_devices":
            total_devices = int(v)
        base_args.append(f"--{arg}={v}")
    return base_device, total_devices, base_args


def start_processes(base_device, total_devices, base_args):

    base_args, total_devices, base_args = parse_args(FLAGS.args)

    processes = []

    for i in range(total_devices):
        sub_base_device = f"{base_device}:{i}"
        p = multiprocessing.Process(target=run_script, args=(sub_base_device, total_devices, base_args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    app.run(start_processes)
