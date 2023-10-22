from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from datasets import Features, Sequence
from dataclasses import dataclass
from typing import Literal, Any

FeatureKey = str | tuple[str]
FeatureMappingT = (
    FeatureKey | list["FeatureMappingT"] | dict[str, "FeatureMappingT"]
)


@dataclass
class FormatFeaturesConfig(BaseDataProcessorConfig):
    """(Re-Format) Dataset Features Processor Config

    Re-Formats Features of the dataset according to the
    specified mapping.

    Type Identifier: `hyped.data.processors.features.format`

    Attributes:
        mapping (dict[str, FeatureMappingT]):
            feature mapping describing the formatted target features,
            Leafs of the (nested) mapping must be valid feature names
            of existing dataset features or paths (i.e. tuple of strings)
            in case of nested features.
    """

    t: Literal[
        "hyped.data.processors.features.format"
    ] = "hyped.data.processors.features.format"

    mapping: dict[str, FeatureMappingT] = None


class FormatFeatures(BaseDataProcessor[FormatFeaturesConfig]):
    """(Re-Format) Dataset Features Processor

    Re-Formats Features of the dataset according to the
    mapping in the config.

    Arguments:
        config (FormatFeaturesConfig): formatting configuration
    """

    def map_features(self, features: Features) -> Features:
        """Apply formatting to given features

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features):
                new features formatted according to the feature
                mapping specified in the config
        """

        def _resolve_feature_from_path(path, _features):
            for i, key in enumerate(path):
                if isinstance(key, int):
                    # check feature type
                    if not isinstance(_features, (Sequence, list)):
                        raise TypeError(
                            "Expected feature type at path `%s` to be a "
                            "`Sequence` (or `list`), got %s"
                            % (str(path[:i]), _features)
                        )
                    # check key
                    if (_features.length >= 0) and (key >= _features.length):
                        raise IndexError(
                            "Index `%i` out of bounds for sequence of "
                            "length `%i` of feature at path %s"
                            % (key, _features.length, str(path[:i]))
                        )
                    # get feature type at index
                    _features = (
                        _features.feature
                        if isinstance(_features, Sequence)
                        else _features[0]
                    )

                elif isinstance(key, str):
                    # check feature type
                    if not isinstance(_features, (Features, dict)):
                        raise TypeError(
                            "Expected feature type at path `%s` to be a "
                            "mapping (i.e. `Features` or `dict`), got %s"
                            % (str(path[:i]), _features)
                        )
                    # check key
                    if key not in _features.keys():
                        raise KeyError(
                            "Key `%s` not present in feature mapping at "
                            "path `%s`, valid keys are %s"
                            % (key, str(path[:i]), list(_features.keys()))
                        )
                    # get feature type at key
                    _features = _features[key]

                else:
                    # unexpected key type in path
                    raise TypeError(
                        "Unexpected key type in path, valid types are `str` "
                        "or `int`, got %s" % key
                    )

            return _features

        def _map_features(mapping):
            if isinstance(mapping, str):
                if mapping not in features:
                    raise KeyError(
                        "Feature `%s` not present in input features, "
                        "valid feature names are %s"
                        % (mapping, str(list(features.keys())))
                    )

                return features[mapping]

            if isinstance(mapping, tuple):
                # check path
                if len(mapping) == 0:
                    raise ValueError(
                        "Found empty path (i.e. tuple of length 0) in "
                        "feature mapping"
                    )
                # resolve
                return _resolve_feature_from_path(mapping, features)

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

        def _resolve_value_from_path(path, _example):
            # follow path
            for key in path:
                # works for both sequences and mappings
                _example = _example[key]

            return _example

        def _map_example(mapping):
            if isinstance(mapping, str):
                return example[mapping]

            if isinstance(mapping, tuple):
                return _resolve_value_from_path(mapping, example)

            if isinstance(mapping, list):
                return list(map(_map_example, mapping))

            if isinstance(mapping, dict):
                return {k: _map_example(v) for k, v in mapping.items()}

        return _map_example(self.config.mapping)
