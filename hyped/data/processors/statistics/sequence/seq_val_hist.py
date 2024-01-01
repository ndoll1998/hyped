import numpy as np
from hyped.data.processors.statistics.value.hist import (
    Histogram,
    HistogramConfig,
)
from hyped.utils.feature_access import (
    get_feature_at_key,
    batch_get_value_at_key,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_is_sequence,
)
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Any, Literal
from datasets import Features
from itertools import chain


# TODO: write tests for sequence value histogram


@dataclass
class SequenceValueHistogramConfig(HistogramConfig):
    """Sequence Value Histogram Data Statistic Config

    Build a histogram of the values of a given sequence feature.

    Type Identifier: "hyped.data.processors.statistics.value.seq_val_histogram"

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
        "hyped.data.processors.statistics.sequence.seq_val_histogram"
    ] = "hyped.data.processors.statistics.sequence.seq_val_histogram"


class SequenceValueHistogram(Histogram):
    """Sequence Value Histogram Data Statistic

    Build a histogram of the values of a given sequence feature.
    """
    # overwrite config type
    CONFIG_TYPE = SequenceValueHistogramConfig

    def check_features(self, features: Features) -> None:
        """Check input features.

        Makes sure the feature key specified in the configuration is present
        in the features and the feature type is a sequence.

        Warns when the sequence feature is fixed-sized.

        Arguments:
            features (Features): input dataset features
        """
        # make sure feature exists and is a sequence
        raise_feature_exists(self.config.feature_key, features)
        feature = get_feature_at_key(features, self.config.feature_key)
        raise_feature_is_sequence(
            self.config.feature_key,
            feature,
        )

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> tuple[NDArray, NDArray]:
        """Compute the sequence value histogram for the given batch
        of examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            bin_ids (NDArray): array of integers containing the bin ids
            bin_counts (NDArray): array of integers containing the bin counts
        """
        x = batch_get_value_at_key(examples, self.config.feature_key)
        x = np.asarray(list(chain.from_iterable(x)))

        return self._compute_histogram(x)
