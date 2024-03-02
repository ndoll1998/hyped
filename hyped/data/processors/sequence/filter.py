from dataclasses import dataclass
from itertools import compress
from typing import Any, Literal

from datasets import ClassLabel, Features, Sequence, Value

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
    get_sequence_length,
    raise_feature_is_sequence,
)


@dataclass
class FilterSequenceConfig(BaseDataProcessorConfig):
    """Filter Sequence Data Processor Config

    Discard all items of a sequence that are not present
    in a specified set of valid values. The processor also
    converts the sequence feature to a `ClassLabel` instance
    which tracks ids instead of the values.

    Attributes:
        sequence (FeatureKey): the key to the sequence to filter
        valids (list[Any]): the list of valid values to keep
    """

    t: Literal[
        "hyped.data.processors.sequence.filter"
    ] = "hyped.data.processors.sequence.filter"

    sequence: FeatureKey = None
    valids: list[Any] = None
    # private member for faster lookup
    _valid_set: set[Any] = None

    def __post_init__(self) -> None:
        self._valid_set = set(self.valids)


class FilterSequence(BaseDataProcessor[FilterSequenceConfig]):
    """Filter Sequence Data Processor Config

    Discard all items of a sequence that are not present
    in a specified set of valid values. The processor also
    converts the sequence feature to a `ClassLabel` instance
    which tracks ids instead of the values.
    """

    @property
    def filtered_sequence_feature(self) -> ClassLabel:
        return self.raw_features["filtered_sequence"].feature

    def map_features(self, features: Features) -> Features:
        # check feature
        sequence = get_feature_at_key(features, self.config.sequence)
        raise_feature_is_sequence(self.config.sequence, sequence)
        # get length of the original sequence
        length = get_sequence_length(sequence)
        # build output features
        return {
            "filter_mask": Sequence(Value("bool"), length=length),
            "filtered_sequence": Sequence(
                ClassLabel(names=self.config.valids)
            ),
        }

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get the sequence from the example dict
        seq = get_value_at_key(example, self.config.sequence)
        # compute the mask and
        mask = list(map(self.config._valid_set.__contains__, seq))
        seq = self.filtered_sequence_feature.str2int(compress(seq, mask))
        # return outputs
        return {"filtered_sequence": seq, "filter_mask": mask}
