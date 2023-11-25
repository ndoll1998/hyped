from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_is_sequence,
    get_sequence_length,
    get_sequence_feature,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from itertools import compress
from datasets import Features, Sequence, Value
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class ApplyMaskConfig(BaseDataProcessorConfig):
    """Apply Mask Data Processor Config

    Apply a given mask onto a set of sequences

    Attributes:
        mask (FeatureKey):
            the feature key refering to the mask to apply
        sequences (dict[str, FeatureKey]):
            Collection of feature keys referring to the sequences
            to which to apply the mask. The mask is applied to each
            features referenced by the given keys. The resulting
            masked sequence will be stored under the dictionary key.
    """

    t: Literal[
        "hyped.data.processors.sequence.apply_mask"
    ] = "hyped.data.processors.sequence.apply_mask"

    mask: FeatureKey = None
    sequences: dict[str, FeatureKey] = None


class ApplyMask(BaseDataProcessor[ApplyMaskConfig]):
    """Apply Mask Data Processor

    Apply a given mask onto a set of sequences
    """

    def map_features(self, features: Features) -> Features:
        """Check mask and sequence features and overwrite sequence
        features

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): sequence features to overwrite
        """
        # check mask feature exists and is a sequence of booleans
        mask = get_feature_at_key(features, self.config.mask)
        raise_feature_is_sequence(self.config.mask, mask, Value("bool"))
        # get the length of the mask
        length = get_sequence_length(mask)

        out_features = Features()
        # check each sequence feature
        for name, key in self.config.sequences.items():
            # make sure it exists and is a sequence
            seq = get_feature_at_key(features, key)
            raise_feature_is_sequence(key, seq)
            # it has to be of the same size as the mask
            if length != get_sequence_length(seq):
                raise TypeError(
                    "Length mismatch between mask sequence `%s` and "
                    "sequence `%s`, got %i != %i"
                    % (
                        self.config.mask,
                        key,
                        length,
                        get_sequence_length(seq),
                    )
                )
            # add sequence feature to output features
            out_features[name] = Sequence(get_sequence_feature(seq))

        # return all collected output features
        return out_features

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Apply processor to an example

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]):
        """

        # get the mask
        mask = get_value_at_key(example, self.config.mask)

        out = {}
        # apply mask to each sequence
        for name, key in self.config.sequences.items():
            seq = get_value_at_key(example, key)
            # check length
            if len(seq) != len(mask):
                raise ValueError(
                    "Length mismatch between mask sequence `%s` and "
                    "sequence `%s`, got %i != %i"
                    % (self.config.mask, key, len(mask), len(seq))
                )
            # apply mask to sequence
            out[name] = list(compress(seq, mask))

        return out
