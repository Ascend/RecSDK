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
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import toml


class MxRecConfig:
    """
    Configuration from `config.toml`.
    """

    def __init__(self, pipes: Dict[str, List[str]]):
        self.pipes = pipes
        self.func_to_pipe = defaultdict(str)
        for pipe_name, event_list in self.pipes.items():
            for event in event_list:
                self.func_to_pipe[event] = pipe_name
        self.pipe_names = [name for name in pipes.keys()]


class MxRecEvent:
    """
    Class to represent an MxRec event.
    """

    def __init__(self, log_line: str, event_name: str, pipe_id: int):
        timestamp_s = get_timestamp(log_line)
        duration_ms = get_duration(log_line, event_name)
        process_id = get_process_id(log_line)
        self.timestamp_start_us = timestamp_s * 1e6 - duration_ms * 1e3
        self.duration_us = duration_ms * 1e3
        self.timestamp_end_us = timestamp_s * 1e6
        self.process_id = process_id
        self.name = event_name
        self.pipe_id = pipe_id


@dataclass
class OpEvent:
    """
    Class to represent an Op event.
    """

    device_id: int
    op_name: str
    op_type: str
    task_type: str
    start_timestamp: float
    duration: float


def extract_mxrec_events(
    log_path: str, config: MxRecConfig
) -> Dict[int, Dict[str, List[MxRecEvent]]]:
    """
    Extracts MxRec events from the log file.

    Args:
        log_path (str): Path to the log file.
        config (MxRecConfig): Dictionary mapping event names to pipe names and other configs.

    Returns:
        Dict[int, Dict[str, List[MxRecEvent]]]: Extracted MxRec events grouped by process ID and pipe.
    """
    events: Dict[int, Dict[str, List[MxRecEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    broken_lines = list()
    event_names = config.func_to_pipe
    pipe_names = config.pipe_names
    pipe_ids = defaultdict(int)
    for i, pipe in enumerate(pipe_names):
        pipe_ids[pipe] = i
    with open(log_path) as log:
        for line in log:
            for name, pipe in filter(lambda item: item[0] in line, event_names.items()):
                try:
                    event = MxRecEvent(line, name, pipe_ids[pipe])
                    events[event.process_id][pipe].append(event)
                except RuntimeError:
                    broken_lines.append(line)
    if broken_lines:
        logging.warning("There are %d broken log lines", len(broken_lines))
        for line in broken_lines:
            logging.warning(line)
    return events


def extract_op_events(op_summary_path: str) -> List[OpEvent]:
    """
    Extracts Op events from the CSV file.

    Args:
        op_summary_path (str): Path to the op summary CSV file.

    Returns:
        List[OpEvent]: List of extracted Op events.
    """
    df = pd.read_csv(op_summary_path)
    return [
        OpEvent(
            row["Device_id"],
            row["Op Name"],
            row["OP Type"],
            row["Task Type"],
            row["Task Start Time(us)"],
            row["Task Duration(us)"],
        )
        for _, row in df.iterrows()
    ]


def get_timestamp(log_line: str) -> float:
    """
    Extracts the timestamp from a log line.

    Args:
        log_line (str): A line from the log file.

    Returns:
        float: The extracted timestamp as a float.
    """
    pattern = r"\[(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}\.\d+)\]"
    match = re.search(pattern, log_line)
    if not match:
        raise RuntimeError(f"there is no time in log: {log_line}")
    date_time_str = match.group(1)
    date_time_format = "%Y/%m/%d %H:%M:%S.%f"
    # Parse the date-time string into a datetime object
    date_time_obj = datetime.strptime(date_time_str, date_time_format)
    # Convert the datetime object to a timestamp
    return date_time_obj.timestamp()


def get_duration(log_line: str, event_name: str) -> float:
    """
    Extracts the duration of an event from a log line.

    Args:
        log_line (str): A line from the log file.
        event_name (str): The name of the event.

    Returns:
        int: The extracted duration in milliseconds.
    """
    pattern = event_name + r".*:\s*(\d+)"
    match = re.search(pattern, log_line)
    if not match:
        raise RuntimeError(f"there is no event: {event_name}, log: {log_line}")
    duration_ms = match.group(1)
    return float(duration_ms)


def get_process_id(log_line: str) -> int:
    """
    Extracts the process ID from a log line.

    Args:
        log_line (str): A line from the log file.

    Returns:
        int: The extracted process ID.
    """
    pattern = r"\[(\d+)\]"
    match = re.search(pattern, log_line)
    if not match:
        raise RuntimeError(f"there is no process_id in log: {log_line}")
    process_id = match.group(1)
    return int(process_id)


def read_mxrec_config() -> MxRecConfig:
    """
    Reads the MxRec configuration from a TOML file.

    Returns:
        MxRecCofig: Configuration class.
    """
    try:
        config = toml.load("config.toml")
        return MxRecConfig(config["mxrec"])
    except toml.TomlDecodeError as e:
        raise RuntimeError("can not load config.toml") from e


@dataclass
class TracingMetaData:
    """
    Class to represent metadata for tracing.
    """

    name: str
    pid: int
    tid: int
    ph: str
    args: Dict[str, Any]


class TracingMxRecEvent:
    """
    Class to represent a traced MxRec event.
    """

    def __init__(self, mxrec_event: MxRecEvent):
        self.name = mxrec_event.name
        self.pid = mxrec_event.process_id
        self.tid = get_fake_tid(self.pid, mxrec_event.pipe_id)
        self.ts = mxrec_event.timestamp_start_us
        self.dur = mxrec_event.duration_us
        self.ph = "X"
        self.args = {}


class TracingOpEvent:
    """
    Class to represent a traced Op event.
    """

    def __init__(self, op_event: OpEvent, tid: int):
        self.name = op_event.op_type
        self.pid = get_op_pid(op_event)
        self.tid = tid
        self.ts = op_event.start_timestamp
        self.dur = op_event.duration
        self.ph = "X"
        self.args = {"Op Name": op_event.op_name}


def get_metadata(processes: List[int], config: MxRecConfig) -> List[TracingMetaData]:
    """
    Generates metadata for tracing processes and threads.

    Args:
        processes (List[int]): List of process IDs.
        config (MxRecConfig): Configuration class.

    Returns:
        List[TracingMetaData]: List of tracing metadata.
    """
    metadata = list()
    pipes = config.pipe_names
    for i, pid in enumerate(processes):
        metadata1 = TracingMetaData(
            "process_name", pid, 0, "M", {"name": f"MxRec process {i}"}
        )
        metadata2 = TracingMetaData(
            "process_sort_index", pid, 0, "M", {"sort_index": i}
        )
        metadata.append(metadata1)
        metadata.append(metadata2)
        for pipe_i, pipe in enumerate(pipes):
            pipe_metadata1 = TracingMetaData(
                "thread_name",
                pid,
                get_fake_tid(pid, pipe_i),
                "M",
                {"name": f"{pipe} {pid}"},
            )
            pipe_metadata2 = TracingMetaData(
                "thread_sort_index",
                pid,
                get_fake_tid(pid, pipe_i),
                "M",
                {"sort_index": pipe_i},
            )
            metadata.append(pipe_metadata1)
            metadata.append(pipe_metadata2)
    return metadata


def get_fake_tid(pid: int, pipe_id: int) -> int:
    """
    Generates a fake thread ID based on process ID and pipe ID.

    Args:
        pid (int): Process ID.
        pipe_id (int): Pipe ID.

    Returns:
        int: Fake thread ID.
    """
    return pid * 10 + pipe_id


def get_op_pid(op_event: OpEvent) -> int:
    """
    Gets the process ID for an Op event.

    Args:
        op_event (OpEvent): An Op event.

    Returns:
        int: Process ID.
    """
    # add 100 avoiding confict with cpu pid(rand_id)
    return 100 + op_event.device_id


def get_op_tracing(path: str) -> Tuple[List[TracingMetaData], List[TracingOpEvent]]:
    """
    Generates tracing data for Op events.

    Args:
        path (str): Path to the directory containing Op event summaries.

    Returns:
        Tuple[List[TracingMetaData], List[TracingOpEvent]]: Metadata and tracing events.
    """
    task_types = defaultdict(int)
    pids = set()
    tids = set()
    metadata = list()
    op_tracing = list()

    def new_process_metadata(pid, device_id):
        metadata1 = TracingMetaData(
            "process_name", pid, 0, "M", {"name": f"NPU {device_id}"}
        )
        metadata2 = TracingMetaData(
            "process_sort_index", pid, 0, "M", {"sort_index": pid}
        )
        return [metadata1, metadata2]

    def new_thread_metadata(pid, tid, name):
        metadata1 = TracingMetaData("thread_name", pid, tid, "M", {"name": f"{name}"})
        metadata2 = TracingMetaData(
            "thread_sort_index", pid, tid, "M", {"sort_index": tid}
        )
        return [metadata1, metadata2]

    for root, _, files in os.walk(path):
        for file in files:
            if (
                root.endswith("mindstudio_profiler_output")
                and file.startswith("op_summary")
                and file.endswith(".csv")
            ):
                file_path = os.path.join(root, file)
                op_events = extract_op_events(file_path)
                for event in op_events:
                    process_id = get_op_pid(event)
                    if process_id not in pids:
                        pids.add(process_id)
                        metadata.extend(
                            new_process_metadata(process_id, event.device_id)
                        )
                    if event.task_type not in task_types:
                        task_id = len(task_types)
                        task_types[event.task_type] = task_id
                    tid = get_fake_tid(process_id, task_types[event.task_type])
                    if tid not in tids:
                        tids.add(tid)
                        metadata.extend(
                            new_thread_metadata(process_id, tid, event.task_type)
                        )
                    op_tracing.append(TracingOpEvent(event, tid))
    return metadata, op_tracing


def main():
    """
    Main function to parse arguments and generate tracing JSON.
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate CPU/NPU fusion tracing json."
    )
    parser.add_argument(
        "--debug_log", help="MxRec DEBUG level log file path.", required=True
    )
    parser.add_argument("--msprof_output", help="msprof output path.", required=False)
    args = parser.parse_args()

    log_path = args.debug_log
    tracing = list()
    try:
        config = read_mxrec_config()
        mxrec_events = extract_mxrec_events(log_path, config)
        tracing.extend(get_metadata(list(mxrec_events.keys()), config))
    except RuntimeError:
        logging.error("Can not read config.toml, it will exit unsuccessfully.")
        exit(1)

    for process in mxrec_events.values():
        for events in process.values():
            tracing.extend([TracingMxRecEvent(event) for event in events])

    msprof_output_path = args.msprof_output
    if msprof_output_path:
        op_metadata, op_tracing = get_op_tracing(msprof_output_path)
        tracing.extend(op_metadata)
        tracing.extend(op_tracing)

    fd = os.open("mxrec_tracing.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    with os.fdopen(fd, "w") as file:
        json.dump(tracing, file, indent=4, default=lambda obj: obj.__dict__)


if __name__ == "__main__":
    main()
