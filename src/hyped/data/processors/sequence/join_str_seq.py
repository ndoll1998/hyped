from dataclasses import dataclass
from typing import Any, Literal

from datasets import Features, Value

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_is_sequence,
)


@dataclass
class JoinStringSequenceConfig(BaseDataProcessorConfig):
    """Join String Sequence Data Processor Config

    Concatenate a sequence of strings creating a new string
    formed by adding a specified delimiter in between every
    pair of adjacent strings.

    Type Identifier: `hyped.data.processors.sequence.join_str_seq`

    Attributes:
        sequence (FeatureKey):
            feature key to the sequence of strings to join
        delimiter (str):
            delimiter to use for joining the string sequence.
            Defaults to whitespace character.
        output (str):
            output feature name. Defaults to `joined_string`
    """

    t: Literal[
        "hyped.data.processors.sequence.join_str_seq"
    ] = "hyped.data.processors.sequence.join_str_seq"

    sequence: FeatureKey = None
    delimiter: str = " "
    output: str = "joined_string"


class JoinStringSequence(BaseDataProcessor[JoinStringSequenceConfig]):
    """Join String Sequence Data Processor Config

    Concatenate a sequence of strings creating a new string
    formed by adding a specified delimiter in between every
    pair of adjacent strings.
    """

    def map_features(self, features: Features) -> Features:
        # make sure the feature exists and is a sequence
        # of strings
        raise_feature_exists(self.config.sequence, features)
        raise_feature_is_sequence(
            self.config.sequence,
            get_feature_at_key(features, self.config.sequence),
            Value("string"),
        )
        # returns a string feature
        return Features({self.config.output: Value("string")})

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get the string sequence and join
        return {
            self.config.output: self.config.delimiter.join(
                get_value_at_key(example, self.config.sequence)
            )
        }
