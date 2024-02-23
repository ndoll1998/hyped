from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import raise_feature_exists
from hyped.utils.feature_access import FeatureKey
from datasets import Features
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class FilterFeaturesConfig(BaseDataProcessorConfig):
    """Filter Features Data Processor Config

    Removes dataset features based on the specified filters,
    i.e. the list of features to keep or remove.

    Make sure to specify exactly one of the filters.

    Type Identifier: `hyped.data.processors.features.format`

    Attributes:
        keep (None | list[FeatureKey]): features to keep
        remove (None | list[FeatureKey]): features to remove

    Raises:
        ValueError: when none of the attributes are specified
        ValueError: when both `keep` and `remove` are specified

    """

    t: Literal[
        "hyped.data.processors.features.filter"
    ] = "hyped.data.processors.features.filter"
    # don't keep input features
    keep_input_features: bool = False
    # feature keys to keep or remove
    keep: None | list[FeatureKey] = None
    remove: None | list[FeatureKey] = None


class FilterFeatures(BaseDataProcessor[FilterFeaturesConfig]):
    """Filter Features Data Processor

    Removes dataset features based on the specified filters,
    i.e. the list of features to keep or remove.
    """

    @property
    def required_feature_keys(self) -> list[FeatureKey]:
        # TODO: when remove is defined this should be input_features \ remove
        if (self.remove is not None):
            raise NotImplementedError()

        return list(self.config.required_feature_keys)

    def map_features(self, features: Features) -> Features:
        """Filter dataset feature mapping

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): filtered dataset features
        """

        keep = self.config.keep
        remove = self.config.remove

        # check configuration
        if (keep is None) and (remove is None):
            raise ValueError(
                "No filters specified, please specify either the `keep` "
                "or `remove` filters in the configuration"
            )

        if (keep is not None) and (remove is not None):
            raise ValueError(
                "Please specify either the `keep` or the `remove` filter "
                "but not both"
            )

        # make sure all features exist
        for k in keep if keep is not None else remove:
            raise_feature_exists(k, features)

            # TODO: currently only supports string keys
            if not isinstance(k, str):
                raise NotImplementedError(
                    "Currently only simple string keys are "
                    "supported by the filter processor"
                )

        if keep is not None:
            # collect features
            return Features({k: features[k] for k in keep})

        if remove is not None:
            # remove features
            remove = set(remove)
            return Features(
                {k: v for k, v in features.items() if k not in remove}
            )

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Filter features in example

        Arguments:
            example (dict[str, Any]): example to filter
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): filtered example
        """
        # collect values of features to keep
        return {k: example[k] for k in self.new_features.keys()}