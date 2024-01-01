from __future__ import annotations
import numpy as np
from datasets import Features
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
from dataclasses import dataclass
from typing import Any, Literal
from math import sqrt


@dataclass
class MeanAndStdTuple(object):
    """Mean and Standard Deviation Container

    Packs the mean and standard deviation values into a statistic
    container.

    Attributes:
        mean (float): mean value
        std (float): standard deviation value
        n (int): total number of samples
    """

    mean: float = 0.0
    std: float = 0.0
    n: int = 0

    def __eq__(self, other) -> bool:
        return (
            (self.n == other.n)
            and abs((self.mean - other.mean) < 1e-8)
            and abs((self.std - other.std) < 1e-8)
        )

    @staticmethod
    def incremental_mean_and_std(
        a: MeanAndStdTuple,
        b: MeanAndStdTuple,
    ) -> MeanAndStdTuple:
        """Implementations for the incremental Mean and Standard Deviation
        formulas. Computes the combined mean and standard deviation of two
        seperate instances.

        Arguments:
            a (MeanAndStdTuple): mean and standard deviation of distribution a
            b (MeanAndStdTuple): mean and standard deviation of distribution b

        Returns:
            s (MeanAndStdTuple):
                mean and standard deviation of joined distribution ab
        """
        # unpack current value and compute values for given batch
        m1, s1, n1 = a.mean, a.std, a.n
        m2, s2, n2 = b.mean, b.std, b.n
        # batch incremental mean and std formulas
        m_new = (m1 * n1 + m2 * n2) / (n1 + n2)
        s_new = sqrt(
            (s1 * s1 * n1 + s2 * s2 * n2) / (n1 + n2)
            + (n1 * n2 * (m1 - m2) ** 2) / ((n1 + n2) ** 2)
        )
        # pack new values
        return MeanAndStdTuple(m_new, s_new, n1 + n2)


@dataclass
class MeanAndStdConfig(BaseDataStatisticConfig):
    """Mean and Standard Deviation Data Statistic Config

    Compute the mean and standard deviation of a given
    feature.

    Type Identifier: "hyped.data.processors.statistics.value.mean_and_std"

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
        feature_key (FeatureKey):
            key to the dataset feature of which to compute the
            mean and standard deviation of
    """

    t: Literal[
        "hyped.data.processors.statistics.value.mean_and_std"
    ] = "hyped.data.processors.statistics.value.mean_and_std"

    feature_key: FeatureKey = None


class MeanAndStd(BaseDataStatistic[MeanAndStdConfig, MeanAndStdTuple]):
    """Mean and Standard Deviation Data Statistic Config

    Compute the mean and standard deviation of a given
    feature.
    """

    def initial_value(self, features: Features) -> MeanAndStdTuple:
        """Initial value for mean and standard deviation statistic

        Arguments:
            features (Features): input dataset features

        Returns:
            init_val (MeanAndStdTuple): inital value of all zeros
        """
        return MeanAndStdTuple()

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
    ) -> MeanAndStdTuple:
        """Extract mean and standard deviation from batch of examples

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            ext (MeanAndStdTuple): mean and standard deviation
        """
        # get batch of values from examples
        x = batch_get_value_at_key(examples, self.config.feature_key)
        x = np.asarray(x)
        # compute mean and standard deviation and pack together
        return MeanAndStdTuple(x.mean(), x.std(), len(x))

    def compute(
        self,
        val: MeanAndStdTuple,
        ext: MeanAndStdTuple,
        index: list[int],
        rank: int,
    ) -> MeanAndStdTuple:
        """Compute updated statistic

        Applies incremental formulas for mean and standard deviation on
        the current statistic value and the extracted mean and standard
        deviation.

        Arguments:
            val (MeanAndStdTuple): current statistic value
            ext (MeanAndStdTuple): extracted mean and standard deviation
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            new_val (MeanAndStdTuple): combined mean and standard deviation
        """
        return MeanAndStdTuple.incremental_mean_and_std(val, ext)
