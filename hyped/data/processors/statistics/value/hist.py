import numpy as np
import multiprocessing as mp
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    batch_get_value_at_key,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_equals,
    INT_TYPES,
    UINT_TYPES,
    FLOAT_TYPES,
)
from hyped.data.processors.statistics.base import (
    BaseDataStatistic,
    BaseDataStatisticConfig,
)
from hyped.data.processors.statistics.report import StatisticsReportStorage
from datasets import Features
from dataclasses import dataclass
from typing import Any, Literal
from numpy.typing import NDArray


@dataclass
class HistogramConfig(BaseDataStatisticConfig):
    """Histogram Data Statistic Config

    Build a histogram of a given value feature.

    Type Identifier: "hyped.data.processors.statistics.value.histogram"

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to build the
            histogram
        low (float): lower end of the range of the histogram
        high (float): upper end of the range of the histogram
        num_bins (int): number of bins of the histogram

    """

    t: Literal[
        "hyped.data.processors.statistics.value.histogram"
    ] = "hyped.data.processors.statistics.value.histogram"

    feature_key: FeatureKey = None
    # histogram range and number of bins
    low: float = None
    high: float = None
    num_bins: int = None


class Histogram(BaseDataStatistic[HistogramConfig, list[int]]):
    """Histogram Data Statistic Config

    Build a histogram of a given value feature.
    """

    def initial_value(
        self, features: Features, manager: mp.Manager
    ) -> list[int]:
        """Initial histogram of all zeros.

        The return value is a list proxy to allow to share
        it between processes instead of copying the whole
        histogram to each process.

        Arguments:
            features (Features): input dataset features

        Returns:
            init_val (list[int]): inital histogram of all zeros
        """
        return manager.list([0] * self.config.num_bins)
    
    def check_features(self, features: Features) -> None:
        """Check input features.

        Makes sure the feature key specified in the configuration is present
        in the features and the feature type is a scalar value.

        Arguments:
            features (Features): input dataset features
        """
        raise_feature_exists(self.config.feature_key, features)
        raise_feature_equals(
            self.config.feature_key,
            get_feature_at_key(features, self.config.feature_key),
            INT_TYPES + UINT_TYPES + FLOAT_TYPES,
        )
    
    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> tuple[NDArray, NDArray]:
        """Compute the histogram for the given batch of examples

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            bin_ids (NDArray): array of integers containing the bin ids
            bin_counts (NDArray): array of integers containing the bin counts
        """
        # get batch of values from examples
        x = batch_get_value_at_key(examples, self.config.feature_key)
        x = np.asarray(x)
        # find bin to each value
        bin_size = (self.config.high - self.config.low) / (self.config.num_bins - 1)
        bins = ((x - self.config.low) // bin_size).astype(np.int32)
        # build histogram for current examples from bins
        return np.unique(bins, return_counts=True)

    def compute(
        self,
        val: list[int],
        ext: tuple[NDArray, NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Compute the total sub-histogram for the current batch of examples.

        The total sub-histogram is the sub-histogram computed by the `extract`
        function with the counts being the total counts computed by adding the
        values of the current histogram.

        Arguments:
            val (list[int]): current histogram
            ext (tuple[NDArray, NDArray]): extracted sub-histogram

        Returns:
            bin_ids (NDArray): array of integers containing the bin ids
            total_bin_counts (NDArray):
                array of integers containing the total bin counts
        """
        # add values of original histogram
        bin_ids, bin_counts = ext
        return bin_ids, bin_counts + np.asarray([val[i] for i in bin_ids])

    def update(self, report: StatisticsReportStorage, val: tuple[NDArray, NDArray]) -> None:
        """Write the new histogram values to the report

        Arguments:
            report (StatisticsReportStorage):
                report storage to update the statistic in
            cal (tuple[NDArray, NDArray]): total sub-histogram
        """
        # get histogram
        hist = report.get(self.config.statistic_key)
        # write new values to histogram
        bin_ids, bin_counts = val
        for i, c in zip(bin_ids, bin_counts):
            hist[i] = c
