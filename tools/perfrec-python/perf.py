#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse
import logging
import os
import subprocess
from collections import defaultdict
from typing import List

import toml
from tabulate import tabulate

NEW_FILE_FLAG = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def generate_flamegraph(
    perf_bin: str, perf_data: str, output_svg: str, flamegraph_path: str
) -> None:
    """
    Generate a flamegraph from perf data.

    Args:
        perf_data (str): Path to the perf.data file.
        output_svg (str): Path to the output SVG file.
        flamegraph_path (str): Path to the Flamegraph scripts directory.
    """
    # Ensure perf script is available
    try:
        subprocess.run([perf_bin, "--version"], shell=False, check=True)
    except subprocess.CalledProcessError:
        logging.error("perf is not installed or not in PATH.")
        return

    # Ensure Flamegraph scripts are available
    stackcollapse_path = os.path.join(flamegraph_path, "stackcollapse-perf.pl")
    flamegraph_script_path = os.path.join(flamegraph_path, "flamegraph.pl")

    if not os.path.isfile(stackcollapse_path) or not os.path.isfile(
        flamegraph_script_path
    ):
        logging.error(
            "Flamegraph scripts not found in the provided directory %s.",
            flamegraph_path,
        )
        return

    # Generate the folded stack output
    folded_output = perf_data + ".folded"
    fd = os.open(folded_output, NEW_FILE_FLAG, 0o640)
    with os.fdopen(fd, "w") as f:
        script_output = subprocess.run(
            [perf_bin, "script", "-i", perf_data],
            shell=False,
            check=True,
            stdout=subprocess.PIPE,
        )
        subprocess.run(
            [stackcollapse_path],
            shell=False,
            check=True,
            input=script_output.stdout,
            stdout=f,
        )

    # Generate the flamegraph
    fd_svg = os.open(output_svg, NEW_FILE_FLAG, 0o640)
    with os.fdopen(fd_svg, "w") as f:
        subprocess.run(
            [flamegraph_script_path, folded_output], shell=False, check=True, stdout=f
        )

    logging.info("Flamegraph generated at %s", output_svg)

    # Analyze the folded stack output
    analyze_folded_stack(folded_output)


class CallStack:
    def __init__(self):
        self.count = 0
        self.call_stacks = []

    def add_call_stacks(self, count: int, call_stack: str):
        self.count += count
        self.call_stacks.append(call_stack)


def analyze_folded_stack(folded_output: str) -> None:
    """
    Analyzes the folded stack output to find functions with significant sample counts.

    Args:
        folded_output (str): Path to the folded stack output file.
    """

    function_counts = defaultdict(CallStack)
    total_count = 0

    # Read the folded stack output
    # Line of folded stack example:
    # python3.7;[libascendalog.so];access;__sys_trace_return;prepare_creds 10101010
    with open(folded_output, "r") as f:
        for line in f:
            parts = line.strip().rsplit(
                " ", 1
            )  # Use rsplit to handle function names with spaces
            count = int(parts[-1])
            call_stack_str = parts[0]
            stack = parts[0].split(";")
            function_counts[stack[-1]].add_call_stacks(count, call_stack_str)
            total_count += count

    config = read_config()

    # Filter and display functions with more than 5% total count
    threshold = total_count * config.threshold
    results = [
        (func, call_stack)
        for func, call_stack in function_counts.items()
        if call_stack.count >= threshold and func not in config.ignores
    ]

    # Sort results by count in descending order
    results.sort(key=lambda x: x[1].count, reverse=True)

    # Prepare data for tabulate
    # Write call stacks to file
    table_data = []
    fd_call_stacks = os.open("call_stacks.txt", NEW_FILE_FLAG, 0o640)
    with os.fdopen(fd_call_stacks, "w") as f:
        for func, call_stack in results:
            percentage = (
                (call_stack.count / total_count) * 100 if total_count != 0 else 0
            )
            table_data.append(
                [limit_line(func, 50), call_stack.count, f"{percentage:.2f}%"]
            )
            stacks = [stk + "\n" for stk in call_stack.call_stacks]
            f.writelines(
                [
                    f"func_name: {func}\n",
                    f"percentage: {percentage:.2f}%\n",
                    "call_stacks:\n",
                ]
                + stacks
                + ["\n\n"]
            )

    # Print the results using tabulate
    logging.info("\nFunctions with more than 5% of total samples:")
    headers = ["Function", "Count", "Percentage"]
    logging.info("\n%s", tabulate(table_data, headers=headers, tablefmt="grid"))


def limit_line(input_content: str, line_length: int) -> str:
    """
    Limits the length of a line to a specified number of characters, adding line breaks if necessary.

    Args:
        input_content (str): The input string.
        line_length (int): The maximum line length.

    Returns:
        str: The formatted string with line breaks.
    """
    if line_length >= len(input_content):
        return input_content
    limited_str = ""
    if line_length > 0:
        limited_str = "\n".join(
            input_content[i: i + line_length]
            for i in range(len(input_content), line_length)
        )
    return limited_str


class PerfConfig:
    """
    Configuration from `config.toml`.
    """

    def __init__(self, ignores: List[str], threshold: float = 0.05):
        self.ignores = set(ignores)
        self.threshold = threshold


def read_config() -> PerfConfig:
    """
    Reads configs related to `perf` from the configuration file.

    Returns:
        PerfConfig: Configuration class.
    """
    try:
        config = toml.load("config.toml")
        perf_config = config["perf"]
        return PerfConfig(perf_config["ignores"], perf_config["threshold"])
    except toml.TomlDecodeError:
        return PerfConfig(ignores=[])


def main():
    """
    Main function to parse arguments and generate a flamegraph.
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate a Flamegraph from perf.data."
    )
    parser.add_argument(
        "--perf_data", help="Path to the perf.data file.", required=True
    )
    parser.add_argument(
        "--flamegraph_path",
        help="Path to the Flamegraph Perl scripts directory.",
        required=True,
    )
    parser.add_argument(
        "--perf_bin",
        help="Path to perf exacutable binary file. (default: perf)",
        required=False,
        default="perf",
    )
    parser.add_argument(
        "--output_svg",
        help="Path to the output SVG file. (default: flamegraph.svg)",
        required=False,
        default="flamegraph.svg",
    )
    args = parser.parse_args()

    generate_flamegraph(
        args.perf_bin, args.perf_data, args.output_svg, args.flamegraph_path
    )


if __name__ == "__main__":
    main()
