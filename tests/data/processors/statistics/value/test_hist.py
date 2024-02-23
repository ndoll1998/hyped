import numpy as np
import pytest
from datasets import Features, Value

from hyped.data.processors.statistics.value.hist import (
    Histogram,
    HistogramConfig,
)
from tests.data.processors.statistics.base import BaseTestDataStatistic


class TestHistogram(BaseTestDataStatistic):
    @pytest.fixture
    def in_features(self):
        return Features({"A": Value("float64")})

    @pytest.fixture(
        params=[
            list(range(10)),
            [0, 0, 0, 1, 1, 1, 3, 3, 3],
            np.random.randint(-10, 10, size=100).tolist(),
            np.random.randint(-10, 10, size=5000).tolist(),
        ]
    )
    def in_batch(self, request):
        return {"A": request.param}

    @pytest.fixture
    def statistic(self, in_batch):
        return Histogram(
            HistogramConfig(
                statistic_key="A.histogram",
                feature_key="A",
                low=min(in_batch["A"]),
                high=max(in_batch["A"]),
                num_bins=max(in_batch["A"]) - min(in_batch["A"]) + 1,
            )
        )

    @pytest.fixture
    def expected_init_value(self, statistic):
        return [0] * statistic.config.num_bins

    @pytest.fixture
    def expected_stat_value(self, in_batch, statistic):
        vals = np.asarray(in_batch["A"]) - statistic.config.low
        return np.bincount(vals, minlength=statistic.config.num_bins).tolist()
