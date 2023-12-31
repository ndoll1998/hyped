from hyped.data.processors.statistics.base import (
    BaseDataStatistic,
    BaseDataStatisticConfig,
)
from hyped.data.processors.statistics.report import (
    StatisticsReport,
)
from datasets import Features, Value
from dataclasses import dataclass
import pytest

from tests.data.processors.statistics.test_report import is_lock_acquired


@dataclass
class ConstantStatisticConfig(BaseDataStatisticConfig):
    statistic_key: str = "constant"
    init_val: int = 0
    val: int = 1


class ConstantStatistic(BaseDataStatistic[ConstantStatisticConfig, int]):
    def __init__(self, config, report):
        super(ConstantStatistic, self).__init__(config)
        self.report = report

    @property
    def lock(self):
        return self.report.storage.get_lock_for(self.config.statistic_key)

    def check_features(self, features):
        return

    def initial_value(self, features):
        return self.config.init_val

    def extract(self, examples, index, rank):
        # test if lock is acquired
        assert not is_lock_acquired(self.lock)
        return "EXTRACTED_VALUE"

    def update(self, val, ext, index, rank):
        # check extracted value and current statistic value
        assert ext == "EXTRACTED_VALUE"
        assert val in {self.config.val, self.initial_value(self.in_features)}
        # make sure lock is acquired for update
        assert is_lock_acquired(self.lock)
        return self.config.val


class TestDataStatistic(object):
    @pytest.fixture
    def report(self):
        return StatisticsReport()

    def test_basic(self, report):
        with report:
            # create statistic processor and make sure key is not registered
            stat = ConstantStatistic(ConstantStatisticConfig(), report)
            assert stat.config.statistic_key not in report

            # prepare statistic and now the key should be registered
            stat.prepare(Features({"A": Value("int32")}))
            assert stat.config.statistic_key in report
            assert report[stat.config.statistic_key] == stat.config.init_val

            batch = {"A": list(range(10))}
            # execute statistic processor
            out_batch = stat.batch_process(
                batch, index=list(range(10)), rank=0, return_index=False
            )
            # check output batch and final statistic value
            assert out_batch == batch
            assert report[stat.config.statistic_key] == stat.config.val
