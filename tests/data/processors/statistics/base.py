from abc import abstractmethod
from multiprocessing.managers import BaseProxy
from typing import Any

import pytest
from datasets import Features

from hyped.data.processors.statistics.base import BaseDataStatistic
from hyped.data.processors.statistics.report import StatisticsReport
from tests.data.processors.base import BaseTestDataProcessor


class BaseTestDataStatistic(BaseTestDataProcessor):
    @pytest.fixture
    @abstractmethod
    def statistic(self, request) -> BaseDataStatistic:
        ...

    @pytest.fixture
    def expected_stat_value(self) -> None | Any:
        return None

    @pytest.fixture
    def expected_init_value(self) -> None | Any:
        return None

    @pytest.fixture
    def report(self) -> StatisticsReport:
        report = StatisticsReport()
        with report:
            yield report
        del report

    @pytest.fixture
    def processor(self, statistic) -> BaseDataStatistic:
        return statistic

    @pytest.fixture
    def expected_out_features(self) -> Features:
        return Features()

    @pytest.fixture
    def expected_out_batch(self) -> dict[str, Any]:
        return {}

    @pytest.fixture(params=[10, 100, 1000])
    def batch_size(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 5])
    def num_proc(self, request):
        return request.param

    @pytest.fixture
    def kwargs_for_post_prepare_checks(
        self, report, statistic, in_features, expected_init_value
    ):
        return {
            "report": report,
            "statistic": statistic,
            "in_features": in_features,
            "expected_init_value": expected_init_value,
        }

    @pytest.fixture
    def kwargs_for_post_process_checks(
        self, report, statistic, expected_stat_value
    ):
        return {
            "report": report,
            "statistic": statistic,
            "expected_stat_value": expected_stat_value,
        }

    def post_prepare_checks(
        self, report, statistic, in_features, expected_init_value
    ):
        # statistics do not generate features
        # thus the expected out features are empty
        super(BaseTestDataStatistic, self).post_prepare_checks(
            statistic, in_features, Features()
        )
        # make sure key is registered
        assert statistic.config.statistic_key in report
        if expected_init_value is not None:
            # get value stored in report
            val = report[statistic.config.statistic_key]
            # get the actual value when the object is a proxy
            if isinstance(val, BaseProxy):
                val = val._getvalue()
            # compare
            assert val == expected_init_value

    def post_process_checks(
        self,
        index,
        batch,
        out_batch,
        report,
        statistic,
        expected_stat_value,
    ):
        # statistics do not generate features
        # this the expected output is the same as the input
        super(BaseTestDataStatistic, self).post_process_checks(
            index, batch, out_batch, statistic, {}
        )
        # test statistic value after execution
        if expected_stat_value is not None:
            # get value stored in report
            val = report[statistic.config.statistic_key]
            # get the actual value when the object is a proxy
            if isinstance(val, BaseProxy):
                val = val._getvalue()
            # compare
            assert val == expected_stat_value

    @pytest.fixture(params=[10, 100, 1000])
    def map_batch_size(self, request):
        return request.param
