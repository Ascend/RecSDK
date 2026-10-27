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

"""Precision-consistency tests: Ascend ``dynamic_emb`` vs the v25.09 reference.

Each case replays one deterministic scenario (CONSTANT init, STEP score, fixed
inputs, no eviction) for one supported optimizer and compares the numbers with
the reference bundle recorded on the NVIDIA side by ``record_reference.py``.

    python3 -m pytest test_precision_consistency.py -v

Cases whose reference result has not been recorded yet are *skipped* with an
explicit message, never silently passed.
"""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backend as be  # noqa: E402
import scenario as sc  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "reference_v25.09.json")

#: Tolerances are a *test-side policy*: the golden bundle only stores numbers.
#: SGD / EXACT_SGD / AdaGrad reproduce bit-identically across CUDA and Ascend
#: (measured 2026-10-21 on A100 vs Ascend950PR), so they keep atol=0.
#: ADAM keeps beta-weighted moment accumulators; fp32 reordering across the two
#: backends shows up as drift (measured max |d loss| 4.3e-3, i.e. 4e-5 relative,
#: and 4e-4 relative on the pooled probe), so it gets a relative tolerance.
TOLERANCE = {
    "ADAM": {"loss_rtol": 1e-3, "digest_rtol": 2e-3},
}
# Measured 2026-10-26 (A100 vs Ascend950PR, DEBUG init): SGD/AdaGrad differ by
# 1 fp32 ULP (relative 8.5e-8 on 1.8e5, 1.1e-7 on 2.8e8), i.e. reduction order,
# not algorithm. 1e-6 leaves one decade of margin above that.
DEFAULT_TOLERANCE = {"loss_rtol": 1e-6, "digest_rtol": 1e-6}
#: absolute floors for the pooled-output digests (float64 sums of float32 values)
DIGEST_ATOL = {"sum": 1e-4, "mean": 1e-6, "abs_sum": 1e-4}


@pytest.fixture(scope="session")
def golden():
    if not os.path.exists(GOLDEN):
        pytest.skip(f"reference bundle missing: {GOLDEN}")
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session", autouse=True)
def dist_group():
    be.init_single_process_dist()
    yield
    be.destroy_dist()


@pytest.mark.parametrize("optimizer", sc.DEFAULT_OPTIMIZERS)
def test_precision_matches_reference(optimizer, golden):
    case = golden["cases"].get(optimizer)
    if case is None:
        pytest.skip(f"{optimizer}: reference not recorded yet (run record_reference.py on the GPU side)")

    cfg = dict(case["config"])
    got = sc.run_scenario(cfg)
    assert "fatal" not in got, f"{optimizer}: scenario failed: {got.get('fatal')}"

    ref, sub = case["result"], sc.compact_result(got)
    assert len(sub["losses"]) == len(ref["losses"])

    tol = TOLERANCE.get(optimizer, DEFAULT_TOLERANCE)
    for round_no, (got_loss, ref_loss) in enumerate(zip(sub["losses"], ref["losses"])):
        diff = abs(got_loss - ref_loss)
        limit = tol["loss_rtol"] * abs(ref_loss)
        assert diff <= limit, (
            f"{optimizer} round {round_no}: loss {got_loss!r} vs reference {ref_loss!r} "
            f"(|diff|={diff:.3e} > rtol*|ref|={limit:.3e})"
        )

    for phase in ("initial_probe", "post_training_probe"):
        for key, atol in DIGEST_ATOL.items():
            diff = abs(sub[phase][key] - ref[phase][key])
            limit = atol + tol["digest_rtol"] * abs(ref[phase][key])
            assert diff <= limit, f"{optimizer} {phase}.{key}: |diff|={diff:.3e} > {limit:.3e}"

    # dump/load has its own case (test_dump_load_roundtrip) so a save/load defect
    # cannot mask the precision verdict.
    assert set(sub["dump_load"]) == {"load_changed_state", "loaded_matches_trained"}


def test_optimizer_matrix_is_complete(golden):
    """Every supported optimizer must have a case; a missing one is a hard gap."""
    missing = [name for name in sc.DEFAULT_OPTIMIZERS if name not in golden["cases"]]
    assert not missing, (
        f"golden bundle is missing supported optimizers: {missing}; re-run record_reference.py on the reference side"
    )


def test_unsupported_optimizer_is_rejected():
    """The library itself must reject every EmbOptimType member it does not implement.

    This drives ``convert_optimizer_type`` directly: asserting on scenario.build_case()
    would only exercise this suite's own whitelist, not the library.
    """
    enum = be.api("EmbOptimType")
    convert = be.resolve("convert_optimizer_type")
    unimplemented = [n for n in dir(enum) if n.isupper() and n not in sc.OPTIMIZER_SPECS]
    assert unimplemented, "backend exposes no unimplemented EmbOptimType member"
    for name in unimplemented:
        with pytest.raises(ValueError, match="Not supported optimizer type"):
            convert(getattr(enum, name))


@pytest.mark.parametrize("optimizer", sc.DEFAULT_OPTIMIZERS)
def test_dump_load_roundtrip(optimizer):
    """Reload must reproduce the trained table (needs no reference bundle)."""
    cfg = {"optimizer": optimizer, "rounds": 2, "batch_size": 8}
    got = sc.run_scenario(cfg)
    assert "fatal" not in got, f"{optimizer}: scenario failed: {got.get('fatal')}"
    dump_load = got["phases"]["dump_load"]
    assert dump_load["load_changed_state"], f"{optimizer}: load did not change the fresh table"
    assert dump_load["loaded_matches_trained"], (
        f"{optimizer}: dump/load does not reproduce the trained state "
        f"(load_changed_state={dump_load['load_changed_state']})"
    )
