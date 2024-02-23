from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_access import FeatureKeyCollection
from datasets import Features
from dataclasses import dataclass, field
from typing import Literal, Any, ClassVar


@dataclass
class FormatFeaturesConfig(BaseDataProcessorConfig):
    """(Re-Format) Dataset Features Processor Config

    Re-Formats Features of the dataset according to the
    specified mapping.

    Type Identifier: `hyped.data.processors.features.format`

    Attributes:
        output_format (dict[str, FeatureMappingT]):
            feature mapping describing the formatted target features,
            Leafs of the (nested) mapping must be valid feature names
            of existing dataset features or paths (i.e. tuples) in case
            of nested features.
    """

    # include output format when parsing for required feature keys
    _IGNORE_KEYS_FROM_FIELDS: ClassVar[list[str]] = []

    t: Literal[
        "hyped.data.processors.features.format"
    ] = "hyped.data.processors.features.format"

    output_format: dict[str, FeatureKeyCollection] = field(
        default_factory=dict
    )


class FormatFeatures(BaseDataProcessor[FormatFeaturesConfig]):
    """(Re-Format) Dataset Features Processor

    Re-Formats Features of the dataset according to the
    mapping in the config.

    Arguments:
        config (FormatFeaturesConfig): formatting configuration
    """

    def map_features(self, features: Features) -> Features:
        """Pass input features through. The actual formatting is
        done by the base data processor."""
        return features

    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Pass through all features requested for reformatting.
        The actual formatting is done by the base data processor
        """
        keys = set(self.raw_features.keys())
        return (
            {k: v for k, v in examples.items() if k in keys},
            list(range(len(index))),
        )
