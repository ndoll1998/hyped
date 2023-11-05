from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_exists,
    raise_feature_is_sequence,
    get_sequence_length,
    get_sequence_feature,
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
        mask (str):
            the feature column name containing the mask to apply
        sequences (str | list[str]):
            the feature column name(s) to apply the mask to
    """

    t: Literal[
        "hyped.data.processors.sequence.apply_mask"
    ] = "hyped.data.processors.sequence.apply_mask"

    mask: str = None
    sequences: str | list[str] = None


class ApplyMask(BaseDataProcessor[ApplyMaskConfig]):
    """Apply Mask Data Processor

    Apply a given mask onto a set of sequences
    """

    @property
    def sequences(self) -> list[str]:
        """Sequence feature names"""
        return (
            [self.config.sequences]
            if isinstance(self.config.sequences, str)
            else self.config.sequences
        )

    def map_features(self, features: Features) -> Features:
        """Check mask and sequence features and overwrite sequence
        features

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): sequence features to overwrite
        """
        # check mask feature exists and is a sequence of booleans
        raise_feature_exists(self.config.mask, features)
        raise_feature_is_sequence(
            self.config.mask, features[self.config.mask], Value("bool")
        )
        # get the length of the mask
        length = get_sequence_length(features[self.config.mask])

        # check each sequence feature
        for seq in self.sequences:
            # make sure it exists and is a sequence
            raise_feature_exists(seq, features)
            raise_feature_is_sequence(seq, features[seq])
            # it has to be of the same size as the mask
            if length != get_sequence_length(features[seq]):
                raise TypeError(
                    "Length mismatch between mask sequence `%s` and "
                    "sequence `%s`, got %i != %i"
                    % (
                        self.config.mask,
                        seq,
                        length,
                        get_sequence_length(features[seq]),
                    )
                )

        # build new feature mapping that overwrites the sequences
        return Features(
            {
                seq: Sequence(get_sequence_feature(features[seq]))
                for seq in self.sequences
            }
        )

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
        mask = example[self.config.mask]

        out = {}
        # apply mask to each sequence
        for seq_name in self.sequences:
            seq = example[seq_name]
            # check length
            if len(seq) != len(mask):
                raise ValueError(
                    "Length mismatch between mask sequence `%s` and "
                    "sequence `%s`, got %i != %i"
                    % (self.config.mask, seq_name, len(mask), len(seq))
                )
            # apply mask to sequence
            out[seq_name] = list(compress(seq, mask))

        return out
