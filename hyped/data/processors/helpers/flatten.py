from hyped.data.processors.helpers.format import (
    FormatFeatures,
    FormatFeaturesConfig,
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

    Attributes:
        to_flatten (None | list[str]):
            dataset features to flatten. By default flattens all features
            present in the source feature mapping
        delimiter (str): delimiter used to join nested keys, defaults to ':'
    """

    t: Literal[
        "hyped.data.processors.helpers.flatten"
    ] = "hyped.data.processors.helpers.flatten"

    to_flatten: None | list[str] = None
    delimiter: str = ":"


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
        # overwriting feature mapping in config
        if self.config.mapping is not None:
            warnings.warn(
                "Feature mapping is not None and will be overwritten",
                UserWarning,
            )

        # make sure all features are present
        if self.config.to_flatten is not None:
            for k in self.config.to_flatten:
                if k not in features:
                    raise KeyError(
                        "Feature `%s` not present in input features, "
                        "valid feature names are %s"
                        % (k, str(list(features.keys())))
                    )

        def _build_feature_paths(_features):
            if isinstance(_features, (dict, Features)):
                # recursivly flatten all features in mapping
                return [
                    (k,) + path
                    for k, v in _features.items()
                    for path in _build_feature_paths(v)
                ]

            # only unpack sequences of fixed length
            if isinstance(_features, Sequence) and (_features.length > 0):
                # add to flat feature mapping
                return [
                    (i,) + path
                    for path in _build_feature_paths(_features.feature)
                    for i in range(_features.length)
                ]

            # all other feature types are considered already flat
            return [tuple()]

        # get features to flatten
        to_flatten = self.config.to_flatten or list(features.keys())
        to_flatten = {k: features[k] for k in to_flatten}
        # build feature mapping
        paths = _build_feature_paths(to_flatten)
        self.config.mapping = Features(
            {self.config.delimiter.join(map(str, p)): p for p in paths}
        )
        # map features
        return super(FlattenFeatures, self).map_features(features)
