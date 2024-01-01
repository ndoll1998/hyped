import multiprocessing as mp
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    batch_get_value_at_key,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_equals,
    check_feature_equals,
    INT_TYPES,
    UINT_TYPES,
)
from hyped.data.processors.statistics.base import (
    BaseDataStatistic,
    BaseDataStatisticConfig,
)
from hyped.data.processors.statistics.report import StatisticsReportStorage
from datasets import Features, ClassLabel, Value
from dataclasses import dataclass
from collections import Counter
from typing import Any, Literal


# TODO: write tests for discrete sequence value histogram


@dataclass
class DiscreteHistogramConfig(BaseDataStatisticConfig):
    """Discrete Histogram Data Statistic Config

    Build a histogram of a given discrete value feature,
    e.g. ClassLabel or string.

    Type Identifier: "hyped.data.processors.statistics.value.disc_histogram"

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to build the
            histogram
    """

    t: Literal[
        "hyped.data.processors.statistics.value.discrete_histogram"
    ] = "hyped.data.processors.statistics.value.discrete_histogram"

    feature_key: FeatureKey = None


class DiscreteHistogram(
    BaseDataStatistic[DiscreteHistogramConfig, dict[Any, int]]
):
    """Histogram Data Statistic

    Build a histogram of a given discrete value feature,
    e.g. ClassLabel or string.
    """

    def _map_values(self, vals: list[Any]) -> list[Any]:
        # get feature
        feature = get_feature_at_key(self.in_features, self.config.feature_key)
        # check if feature is a class label
        if check_feature_equals(feature, ClassLabel):
            # map class ids to names
            return feature.int2str(vals)

        return vals

    def _compute_histogram(self, x: list[Any]) -> dict[Any, int]:
        return dict(Counter(self._map_values(x)))

    def initial_value(
        self, features: Features, manager: mp.Manager
    ) -> dict[Any, int]:
        """Initial histogram

        The return value is a dict proxy to allow to share
        it between processes instead of copying the whole
        histogram to each process.

        Arguments:
            features (Features): input dataset features

        Returns:
            init_val (dict[Any, int]): inital histogram dictionary
        """
        return manager.dict()

    def check_features(self, features: Features) -> None:
        """Check input features.

        Makes sure the feature key specified in the configuration is present
        in the features and the feature type is a discrete scalar value.

        Arguments:
            features (Features): input dataset features
        """
        raise_feature_exists(self.config.feature_key, features)
        raise_feature_equals(
            self.config.feature_key,
            get_feature_at_key(features, self.config.feature_key),
            INT_TYPES + UINT_TYPES + [ClassLabel, Value("string")],
        )

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> dict[Any, int]:
        """Compute the histogram for the given batch of examples

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            hist (dict[Any, str]): histogram of given batch of examples
        """
        x = batch_get_value_at_key(examples, self.config.feature_key)
        return self._compute_histogram(x)

    def compute(
        self,
        val: dict[Any, int],
        ext: dict[Any, int],
    ) -> dict[Any, int]:
        """Compute the total sub-histogram for the current batch of examples.

        The total sub-histogram is the sub-histogram computed by the `extract`
        function with the counts being the total counts computed by adding the
        values of the current histogram.

        Arguments:
            val (dict[Any, int]): current histogram
            ext (dicz[Any, int]): extracted sub-histogram

        Returns:
            total_hist (dict[Any, str]): total sub-histogram of current batch
        """
        return {k: v + val.get(k, 0) for k, v in ext.items()}

    def update(
        self, report: StatisticsReportStorage, val: dict[Any, int]
    ) -> None:
        """Write the new histogram values to the report

        Arguments:
            report (StatisticsReportStorage):
                report storage to update the statistic in
            val (dict[Any, int]): total sub-histogram
        """
        # get histogram
        hist = report.get(self.config.statistic_key)
        # write new values to histogram
        for k, c in val.items():
            hist[k] = c
