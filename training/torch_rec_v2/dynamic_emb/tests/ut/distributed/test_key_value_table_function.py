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

import pytest
import torch

from dynamic_emb.distributed.key_value_table import KeyValueTableFunction
from key_value_table_test_helpers import (
    DEFAULT_EMB_DIM,
    EVAL_INITIALIZER,
    TRAIN_INITIALIZER,
    create_training_storage,
    insert_embeddings,
    npu_device,
    read_embeddings_with_founds,
    sample_stored_keys_and_embs,
    sample_update_grads,
    sample_update_keys_and_embs,
)


@pytest.mark.parametrize(
    ("training", "initializer"),
    [
        (True, TRAIN_INITIALIZER),
        (False, EVAL_INITIALIZER),
    ],
)
def test_key_value_table_lookup(training: bool, initializer):
    device = npu_device()
    storage, _ = create_training_storage()

    stored_keys, stored_embs = sample_stored_keys_and_embs(device)
    insert_embeddings(storage, stored_keys, stored_embs)

    unique_keys = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device=device)
    unique_embs = torch.zeros(4, DEFAULT_EMB_DIM, dtype=torch.float32, device=device)

    KeyValueTableFunction.lookup(
        storage=storage,
        unique_keys=unique_keys,
        unique_embs=unique_embs,
        initializer=initializer,
        training=training,
    )

    assert unique_embs.shape == (4, DEFAULT_EMB_DIM)
    torch.testing.assert_close(unique_embs[:2], stored_embs)

    missing_keys = torch.tensor([30, 40], dtype=torch.int64, device=device)
    _, missing_founds = read_embeddings_with_founds(storage, missing_keys)
    if training:
        assert not torch.all(unique_embs[2:] == 0)
        assert missing_founds.all()
    else:
        torch.testing.assert_close(
            unique_embs[2:],
            torch.zeros(2, DEFAULT_EMB_DIM, device=device),
        )
        assert not missing_founds.any()


def test_key_value_table_update():
    device = npu_device()
    storage, optimizer = create_training_storage()

    keys, init_embs = sample_update_keys_and_embs(device)
    insert_embeddings(storage, keys, init_embs)

    unique_grads = sample_update_grads(keys.numel(), DEFAULT_EMB_DIM, device)
    KeyValueTableFunction.update(
        storage=storage,
        unique_keys=keys,
        unique_grads=unique_grads,
        optimizer=optimizer,
    )

    updated_embs, founds = read_embeddings_with_founds(storage, keys)
    assert founds.all()
    assert not torch.allclose(updated_embs, init_embs)


def test_key_value_table_lookup_invalid_dim():
    storage, _ = create_training_storage()
    device = npu_device()

    invalid_keys = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device=device)
    unique_embs = torch.randn(2, DEFAULT_EMB_DIM, device=device)

    with pytest.raises(RuntimeError, match="unique_keys dim not equal 1"):
        KeyValueTableFunction.lookup(storage, invalid_keys, unique_embs, TRAIN_INITIALIZER, training=True)


def test_key_value_table_lookup_empty_keys():
    storage, _ = create_training_storage()
    device = npu_device()

    empty_keys = torch.tensor([], dtype=torch.int64, device=device)
    unique_embs = torch.empty(0, DEFAULT_EMB_DIM, device=device)

    KeyValueTableFunction.lookup(storage, empty_keys, unique_embs, TRAIN_INITIALIZER, training=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
