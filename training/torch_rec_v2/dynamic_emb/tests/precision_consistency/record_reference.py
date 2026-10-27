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

"""Record the reference-side precision results consumed by the pytest cases.

Run this on the **reference** implementation (NVIDIA ``recsys-examples`` v25.09,
CUDA GPU) and commit the produced JSON.  The test cases then replay the same
deterministic scenario on Ascend and compare against it.

    DYNAMICEMB_BACKEND=gpu python3 record_reference.py \
        --out golden/reference_v25.09.json

Re-record only when the reference version or the scenario configuration
changes; the header records what was used.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backend as be  # noqa: E402
import scenario as sc  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "golden", "reference_v25.09.json")
LOG = logging.getLogger("record_reference")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--optimizers", nargs="*", default=sc.DEFAULT_OPTIMIZERS)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    be.init_single_process_dist()
    payload = {
        "schema": 2,
        "reference": {
            "backend": be.BACKEND,
            "upstream": "nvidia/recsys-examples tag v25.09 (corelib/dynamicemb)",
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(be.device()),
        },
        "cases": {},
    }

    for name in args.optimizers:
        cfg = {"optimizer": name, "rounds": args.rounds, "batch_size": args.batch_size}
        res = sc.run_scenario(cfg)
        if res.get("fatal"):
            # never silently skip a case: a partial bundle would look complete
            raise RuntimeError(f"{name}: scenario failed: {res['fatal']}")
        loss = [round_["loss"] for round_ in res["phases"]["training"]]
        LOG.info(
            "%s: losses=%s loaded_matches_trained=%s",
            name,
            loss,
            res["phases"]["dump_load"]["loaded_matches_trained"],
        )
        payload["cases"][name] = {
            "config": cfg,
            "result": sc.compact_result(res),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # compact one-object-per-line layout keeps the committed bundle reviewable
    # without spending hundreds of lines of the PR budget on indentation
    body = ",\n".join(
        f'  {json.dumps(name)}: {{\n'
        f"   \"config\": {json.dumps(case['config'])},\n"
        f"   \"result\": {json.dumps(case['result'])}\n"
        f"  }}"
        for name, case in payload["cases"].items()
    )
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "{\n"
            f" \"schema\": {payload['schema']},\n"
            f" \"reference\": {json.dumps(payload['reference'])},\n"
            " \"cases\": {\n"
            f"{body}\n"
            " }\n"
            "}\n"
        )
    LOG.info("wrote %s (%d cases)", args.out, len(payload["cases"]))
    be.destroy_dist()
    return 0


if __name__ == "__main__":
    sys.exit(main())
