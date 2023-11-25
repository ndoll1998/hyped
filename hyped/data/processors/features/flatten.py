from hyped.data.processors.features.format import (
    FormatFeatures,
    FormatFeaturesConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    get_sequence_length,
    get_sequence_feature,
)
from hyped.utils.feature_access import (
    FeatureKey,
    raise_is_simple_key,
    get_feature_at_key,
    get_value_at_key,
    build_collection_from_keys,
    iter_keys_in_features,
    key_cutoff_at_slice
)
from datasets import Features, Sequence
from dataclasses import dataclass
from typing import Literal
import warnings


@dataclass
class FlattenFeaturesConfig(FormatFeaturesConfig):
    """Flatten Dataset Features Processor Config

    Similar to formatting features (see `hyped.data.processors.helpers.format`)
    but flattens nested features

    Type Identifier: `hyped.data.processors.features.flatten`

    Attributes:
        to_flatten (None | list[str]):
            dataset features to flatten. By default flattens all features
            present in the source feature mapping
        delimiter (str): delimiter used to join nested keys, defaults to ':'
        max_seq_length_to_unpack (int):
            upper threshold of length to unpack sequences. If the sequence
            length exceeds this threshold, the sequence will not be unpacked
    """

    t: Literal[
        "hyped.data.processors.features.flatten"
    ] = "hyped.data.processors.features.flatten"

    to_flatten: None | list[FeatureKey] = None
    delimiter: str = ":"
    max_seq_length_to_unpack: int = 8


class FlattenFeatures(FormatFeatures):
    """Flatten Dataset Features Processor

    Similar to formatting features (see `hyped.data.processors.helpers.format`)
    but flattens nested features

    Arguments:
        config (None | FlattenFeaturesConfig):
            flattening configuration, defaults to flatten all features
    """

    # overwrite config type
    CONFIG_TYPE = FlattenFeaturesConfig

    def __init__(self, config: None | FlattenFeaturesConfig = None) -> None:
        super(FlattenFeatures, self).__init__(
            config=config or FlattenFeaturesConfig()
        )

    def map_features(self, features: Features) -> Features:
        """Flatten dataset feature mapping

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): flattened feature mapping
        """
        # overwriting feature mapping in config
        if self.config.output_format is not None:
            warnings.warn(
                "Feature mapping is not None and will be overwritten",
                UserWarning,
            )

        if self.config.to_flatten is not None:
            # make sure all keys are simple
            for k in self.config.to_flatten:
                raise_is_simple_key(k)

        # get features to flatten, default to all features
        # in the feature mapping
        to_flatten = (
            self.config.to_flatten
            if self.config.to_flatten is not None
            else list(features.keys())
        )

        to_flatten = [
            k if isinstance(k, tuple) else (k,)
            for k in to_flatten
        ]

        collection = build_collection_from_keys(to_flatten)

        for key in to_flatten:
            # get the feature to flatten and the key collection
            # to add the flattened features into
            feature = get_feature_at_key(features, key)
            sub_collection = get_value_at_key(collection, key[:-1])
            assert key[-1] in sub_collection
            # flatten features
            flat_collection = {
                self.config.delimiter.join(map(str, k)): key[:-1] + k
                for k in map(
                    (key[-1],).__add__,
                    map(
                        # TODO: can we flatten after slice?
                        key_cutoff_at_slice,
                        iter_keys_in_features(feature)
                    )
                )
            }

            sub_collection.pop(key[-1])
            sub_collection.update(flat_collection)

        self.config.output_format = collection
        return super(FlattenFeatures, self).map_features(features)
