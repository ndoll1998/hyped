from tests.data.processors.statistics.base import BaseTestDataStatistic
from hyped.data.processors.statistics.value.mean_and_std import (
    MeanAndStd,
    MeanAndStdConfig,
    MeanAndStdTuple,
)
from datasets import Features, Value
import numpy as np
import pytest


class TestMeanAndStd(BaseTestDataStatistic):
    @pytest.fixture
    def in_features(self):
        return Features({"A": Value("float32")})

    @pytest.fixture(
        params=[
            list(range(10)),
            list(range(100)),
            np.random.uniform(0, 1, size=100).tolist(),
            np.random.uniform(0, 1, size=5000).tolist(),
        ]
    )
    def batch(self, request):
        return {"A": request.param}

    @pytest.fixture
    def statistic(self):
        return MeanAndStd(
            MeanAndStdConfig(statistic_key="A.mean_and_std", feature_key="A")
        )

    @pytest.fixture
    def expected_init_value(self):
        return MeanAndStdTuple()

    @pytest.fixture
    def expected_stat_value(self, batch):
        return MeanAndStdTuple(
            mean=np.mean(batch["A"]), std=np.std(batch["A"]), n=len(batch["A"])
        )
