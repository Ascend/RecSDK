import math
import pytest
from torchrec_embcache.distributed.configs import AdmitAndEvictConfig


def test_admit_and_evict_config():
    """DT: 测试AdmitAndEvictConfig的参数校验"""

    # 测试默认值
    config = AdmitAndEvictConfig()
    assert config.admit_threshold == -1
    assert math.isclose(config.not_admitted_default_value, 0.0)
    assert config.evict_threshold == 0
    assert config.evict_step_interval == 0

    # 测试admit_threshold参数
    config = AdmitAndEvictConfig(admit_threshold=5)
    assert config.admit_threshold == 5
    assert config.is_feature_admit_enabled() is True

    # 测试not_admitted_default_value参数
    config = AdmitAndEvictConfig(not_admitted_default_value=0.5)
    assert math.isclose(config.not_admitted_default_value, 0.5)

    # 测试evict_threshold参数
    config = AdmitAndEvictConfig(evict_threshold=10)
    assert config.evict_threshold == 10
    assert config.is_feature_evict_enabled() is True

    # 测试evict_step_interval参数
    config = AdmitAndEvictConfig(evict_step_interval=2)
    assert config.evict_step_interval == 2

    # 测试类型校验
    with pytest.raises(TypeError):
        AdmitAndEvictConfig(admit_threshold="invalid")

    with pytest.raises(TypeError):
        AdmitAndEvictConfig(not_admitted_default_value="invalid")

    with pytest.raises(TypeError):
        AdmitAndEvictConfig(evict_threshold="invalid")

    with pytest.raises(TypeError):
        AdmitAndEvictConfig(evict_step_interval="invalid")

    # 测试值范围校验
    with pytest.raises(ValueError):
        AdmitAndEvictConfig(admit_threshold=-2)

    with pytest.raises(ValueError):
        AdmitAndEvictConfig(evict_threshold=-1)

    with pytest.raises(ValueError):
        AdmitAndEvictConfig(evict_step_interval=-1)
