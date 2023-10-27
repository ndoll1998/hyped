from hyped.data.processors.features.format import (
    FormatFeatures,
    FormatFeaturesConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    get_sequence_length,
    get_sequence_feature,
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

    to_flatten: None | list[str] = None
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
        if self.config.mapping is not None:
            warnings.warn(
                "Feature mapping is not None and will be overwritten",
                UserWarning,
            )

        # make sure all features are present
        if self.config.to_flatten is not None:
            for k in self.config.to_flatten:
                raise_feature_exists(k, features)

        def _build_feature_paths(_features):
            if isinstance(_features, (dict, Features)):
                # recursivly flatten all features in mapping
                return [
                    (k,) + path
                    for k, v in _features.items()
                    for path in _build_feature_paths(v)
                ]

            # only unpack sequences of fixed length
            if isinstance(_features, Sequence):
                length = get_sequence_length(_features)

                if 0 < length < self.config.max_seq_length_to_unpack:
                    # add to flat feature mapping
                    return [
                        (i,) + path
                        for path in _build_feature_paths(
                            get_sequence_feature(_features)
                        )
                        for i in range(length)
                    ]

            # all other feature types are considered already flat
            return [tuple()]

        # get features to flatten
        to_flatten = (
            self.config.to_flatten
            if self.config.to_flatten is not None
            else list(features.keys())
        )
        to_flatten = {k: features[k] for k in to_flatten}
        # build feature mapping
        paths = _build_feature_paths(to_flatten)
        self.config.mapping = Features(
            {self.config.delimiter.join(map(str, p)): p for p in paths}
        )
        # map features
        return super(FlattenFeatures, self).map_features(features)
