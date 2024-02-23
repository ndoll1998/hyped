import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from datasets import Features
from numpy.typing import NDArray

from hyped.data.processors.statistics.value.hist import (
    Histogram,
    HistogramConfig,
)
from hyped.utils.feature_access import (
    batch_get_value_at_key,
    get_feature_at_key,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_is_sequence,
)

# TODO: write tests for sequence length histogram


@dataclass
class SequenceLengthHistogramConfig(HistogramConfig):
    """Sequence Length Histogram Data Statistic Config

    Build a histogram over the lengths of a given sequence feature.

    Type Identifier: "hyped.data.processors.statistics.value.seq_len_histogram"

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to build the
            histogram
        max_length (int):
            maximum sequence length, sequences exceeding this
            threshold will be truncated for the statistics computation
    """

    t: Literal[
        "hyped.data.processors.statistics.sequence.seq_len_histogram"
    ] = "hyped.data.processors.statistics.sequence.seq_len_histogram"

    max_length: int = None

    def __post_init__(self):
        # set values
        self.low = 0
        self.high = self.max_length
        self.num_bins = self.max_length
        # call super function
        super(SequenceLengthHistogramConfig, self).__post_init__()


class SequenceLengthHistogram(Histogram):
    """Sequence Length Histogram Data Statistic

    Build a histogram over the lengths of a given sequence feature.
    """

    # overwrite config type
    CONFIG_TYPE = SequenceLengthHistogramConfig

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
        raise_feature_is_sequence(self.config.feature_key, feature)
        # warn about fixed length sequences
        if feature.length != -1:
            warnings.warn(
                "Computing sequence length histogram of fixed length sequence",
                UserWarning,
            )

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> tuple[NDArray, NDArray]:
        """Compute the sequence length histogram for the given
        batch of examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            bin_ids (NDArray): array of integers containing the bin ids
            bin_counts (NDArray): array of integers containing the bin counts
        """
        x = batch_get_value_at_key(examples, self.config.feature_key)
        lengths = list(map(len, x))

        return self._compute_histogram(np.asarray(lengths))
