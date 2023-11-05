from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_is_sequence,
    raise_object_matches_feature,
    get_sequence_feature,
    get_sequence_length,
)
from datasets import Features, Sequence
from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class UpdateSequenceConfig(BaseDataProcessorConfig):
    """Update Sequence Data Processor Config

    Update a sequence feature by appending or prepending
    new values.

    Arguments:
        sequence (str): the sequence to update
        append (list[Any]): values to append to the sequence
        prepend (list[Any]): values to prepend to the sequence
    """

    t: Literal[
        "hyped.data.processors.sequence.update"
    ] = "hyped.data.processors.sequence.update"

    sequence: str = None
    append: list[Any] = field(default_factory=list)
    prepend: list[Any] = field(default_factory=list)


class UpdateSequence(BaseDataProcessor[UpdateSequenceConfig]):
    """Update Sequence Data Processor Config

    Update a sequence feature by appending or prepending
    new values.
    """

    def map_features(self, features: Features) -> Features:
        # check feature
        raise_feature_exists(self.config.sequence, features)
        raise_feature_is_sequence(
            self.config.sequence, features[self.config.sequence]
        )
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
        return Features(
            {self.config.sequence: Sequence(feature, length=length)}
        )

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get sequence and add new values
        sequence = example[self.config.sequence]
        sequence = self.config.prepend + list(sequence) + self.config.append
        # return updated sequence
        return {self.config.sequence: sequence}
