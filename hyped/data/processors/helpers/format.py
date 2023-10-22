from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from datasets import Features, Sequence
from dataclasses import dataclass
from typing import Literal, Any

FeatureMappingT = str | list["FeatureMappingT"] | dict[str, "FeatureMappingT"]


@dataclass
class FormatFeaturesConfig(BaseDataProcessorConfig):
    """(Re-Format) Dataset Features Processor Config

    Re-Formats Features of the dataset according to the
    specified mapping.

    Type Identifier: `hyped.data.processors.helpers.format`

    Attributes:
        mapping (dict[str, FeatureMappingT]):
            feature mapping describing the formatted target features,
            Leafs of the (nested) mapping must be valid feature names
            of existing dataset features
    """

    t: Literal["hyped.data.processors.helpers.format"]

    mapping: dict[str, FeatureMappingT] = None


class FormatFeatures(BaseDataProcessor[FormatFeaturesConfig]):
    """(Re-Format) Dataset Features Processor

    Re-Formats Features of the dataset according to the
    mapping in the config.

    Arguments:
        config (FormatConfig): formatting configuration
    """

    def map_features(self, features: Features) -> Features:
        """Apply formatting to given features.

        Arguments:
            in (Features): input dataset features

        Returns:
            out (Features):
                new features formatted according to the feature
                mapping specified in the config
        """

        def _map_features(mapping):
            if isinstance(mapping, str):
                if mapping not in features:
                    raise ValueError(
                        "Feature `%s` not present in input features, "
                        "valid feature names are %s"
                        % (mapping, str(list(features.keys())))
                    )

                return features[mapping]

            if isinstance(mapping, list):
                new_features = list(map(_map_features, mapping))
                # make sure all sub-feature-types are equal
                if any(f != new_features[0] for f in new_features[1:]):
                    raise TypeError(
                        "Expected all items of a sequence to be of the "
                        "same feature type, got %s" % str(new_features)
                    )
                # pack into sequence
                return Sequence(
                    feature=new_features[0], length=len(new_features)
                )

            if isinstance(mapping, dict):
                return {k: _map_features(v) for k, v in mapping.items()}

            # unexpected type in mapping
            raise TypeError(
                "Unexpected feature mapping type, allowed types are str, "
                "list and dict, got %s" % str(features)
            )

        # apply mapping to features
        return Features(_map_features(self.config.mapping))

    def process(self, example: dict[str, Any], index: int, rank: int):
        """(re-)format example according to mapping

        Arguments:
            example (dict[str, Any]): example to format
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): formatted example
        """

        def _map_example(mapping):
            if isinstance(mapping, str):
                return example[mapping]

            if isinstance(mapping, list):
                return list(map(_map_example, mapping))

            if isinstance(mapping, dict):
                return {k: _map_example(v) for k, v in mapping.items()}

        return _map_example(self.config.mapping)
