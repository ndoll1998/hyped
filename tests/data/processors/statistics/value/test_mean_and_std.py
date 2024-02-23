import numpy as np
import pytest
from datasets import Features, Value

from hyped.data.processors.statistics.value.mean_and_std import (
    MeanAndStd,
    MeanAndStdConfig,
    MeanAndStdTuple,
)
from tests.data.processors.statistics.base import BaseTestDataStatistic


class TestMeanAndStd(BaseTestDataStatistic):
    @pytest.fixture
    def in_features(self):
        return Features({"A": Value("float64")})

    @pytest.fixture(
        params=[
            list(range(10)),
            list(range(100)),
            np.random.uniform(0, 1, size=100).tolist(),
            np.random.uniform(0, 1, size=5000).tolist(),
        ]
    )
    def in_batch(self, request):
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
    def expected_stat_value(self, in_batch):
        return MeanAndStdTuple(
            mean=np.mean(in_batch["A"]),
            std=np.std(in_batch["A"]),
            n=len(in_batch["A"]),
        )

    @pytest.mark.parametrize("size", [10, 100, 1000])
    @pytest.mark.parametrize("cutoff", [0.2, 0.5, 0.8])
    @pytest.mark.parametrize("seed", [42, 1337, 314159])
    def test_incremental_formulas(self, size, cutoff, seed):
        # create random array
        np.random.seed(seed)
        x = np.random.uniform(size=size)
        # separate into sub-arrays
        n = int(size * cutoff)
        x1, x2 = x[:n], x[n:]
        # compute all means and standard deviations
        mean_and_std = MeanAndStdTuple(x.mean(), x.std(), size)
        mean_and_std_A = MeanAndStdTuple(x1.mean(), x1.std(), n)
        mean_and_std_B = MeanAndStdTuple(x2.mean(), x2.std(), size - n)
        # compare direct vs incremental formulas
        assert mean_and_std == MeanAndStdTuple.incremental_mean_and_std(
            mean_and_std_A,
            mean_and_std_B,
        )
