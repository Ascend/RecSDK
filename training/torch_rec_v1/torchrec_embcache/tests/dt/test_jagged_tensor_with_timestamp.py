import pytest
import torch
from torchrec_embcache.sparse.jagged_tensor_with_timestamp import (
    JaggedTensorWithTimestamp,
    KeyedJaggedTensorWithTimestamp
)


def _make_base_kwargs(timestamps=True):
    """生成共用的 base_kwargs"""
    base = dict(
        values=torch.tensor([1, 2, 3, 4, 5]),
        weights=None,
        lengths=torch.tensor([2, 3]),
        offsets=torch.tensor([0, 2, 5]),
        timestamps=torch.tensor([10, 20, 30, 40, 50]) if timestamps else None
    )
    return base


def _test_common_functionality(cls, extra_kwargs=None):
    """
    通用测试逻辑
    """
    extra_kwargs = extra_kwargs or {}
    base_kwargs = _make_base_kwargs(timestamps=True)
    kwargs = {**base_kwargs, **extra_kwargs}

    # 不提供 timestamps
    no_ts_kwargs = {**kwargs, "timestamps": None}
    instance_no_ts = cls(**no_ts_kwargs)
    assert instance_no_ts.timestamps is None

    # 形状不一致校验（timestamps 长度 ≠ values）
    bad_kwargs = {**kwargs, "timestamps": torch.tensor([1, 2])}
    with pytest.raises(ValueError):
        cls(**bad_kwargs)

    # 类型校验
    invalid_cases = [
        {"values": "invalid"},
        {"weights": "invalid"},
        {"lengths": "invalid"},
        {"offsets": "invalid"},
        {"timestamps": "invalid"},
    ]
    for case in invalid_cases:
        with pytest.raises(TypeError):
            cls(**{**kwargs, **case})


def test_jagged_tensor_with_timestamp():
    _test_common_functionality(JaggedTensorWithTimestamp)


def test_keyed_jagged_tensor_with_timestamp():
    _test_common_functionality(
        KeyedJaggedTensorWithTimestamp,
        extra_kwargs={"keys": ["feat1", "feat2"]}
    )