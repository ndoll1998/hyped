from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_is_sequence,
    raise_object_matches_feature,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from datasets import Features, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal, Any


@dataclass
class ExtendSequenceConfig(BaseDataProcessorConfig):
    """Extend Sequence Data Processor Config

    Extend a sequence feature by appending or prepending
    new values.

    Arguments:
        sequence (FeatureKey): the key to the sequence to extend
        output (str): the output feature name
        append (list[Any]): values to append to the sequence
        prepend (list[Any]): values to prepend to the sequence
    """

    t: Literal[
        "hyped.data.processors.sequence.extend"
    ] = "hyped.data.processors.sequence.extend"

    sequence: FeatureKey = None
    output: str = "output"
    append: list[Any] = field(default_factory=list)
    prepend: list[Any] = field(default_factory=list)


class ExtendSequence(BaseDataProcessor[ExtendSequenceConfig]):
    """Extend Sequence Data Processor Config

    Extend a sequence feature by appending or prepending
    new values.
    """

    def map_features(self, features: Features) -> Features:
        # check feature
        sequence = get_feature_at_key(features, self.config.sequence)
        raise_feature_is_sequence(self.config.sequence, sequence)
        # get item feature type and length of the sequence
        sequence = features[self.config.sequence]
        feature = get_sequence_feature(sequence)
        length = get_sequence_length(sequence)
        # make sure append and prepend values match the feature type
        raise_object_matches_feature(self.config.prepend, Sequence(feature))
        raise_object_matches_feature(self.config.append, Sequence(feature))

        # compute the new sequence length
        if length != -1:
            length += len(self.config.prepend)
            length += len(self.config.append)

        # overwrite sequence feature with new sequence feature with
        # potential new length
        return Features({self.config.output: Sequence(feature, length=length)})

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get sequence and add new values
        sequence = get_value_at_key(example, self.config.sequence)
        sequence = chain(self.config.prepend, sequence, self.config.append)
        # return updated sequence
        return {self.config.output: list(sequence)}
