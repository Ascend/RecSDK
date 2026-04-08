#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import List, Optional, Union, cast


def tabulate(
    table: List[List[Union[str, int]]],
    headers: Optional[List[str]] = None,
    sub_headers: bool = False,
) -> str:
    """
    Format a table as a string.
    Parameters:
        table (list of lists or list of tuples): The data to be formatted as a table.
        headers (list of strings, optional): The column headers for the table. If not provided, the first row of the table will be used as the headers.
    Returns:
        str: A string representation of the table.
    """
    if not table:
        return ""
    if headers is None:
        headers = table[0]
        table = table[1:]
    headers = cast(List[str], headers)
    rows = []
    # Determine the maximum width of each column
    col_widths = [max([len(str(item)) for item in column]) for column in zip(*table)]
    col_widths = [max(i, len(j)) for i, j in zip(col_widths, headers)]
    # Format each row of the table
    for row in table:
        row_str = " | ".join(
            [str(item).ljust(width) for item, width in zip(row, col_widths)]
        )
        rows.append(row_str)
    # Add the header row and the separator line
    rows.insert(
        0,
        " | ".join(
            [header.center(width) for header, width in zip(headers, col_widths)]
        ),
    )

    rows.insert(1, " | ".join(["-" * width for width in col_widths]))
    if sub_headers:
        rows.insert(3, " | ".join(["-" * width for width in col_widths]))
    return "\n".join(rows)