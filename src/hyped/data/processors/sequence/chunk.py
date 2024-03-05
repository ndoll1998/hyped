from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

from datasets import Features

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
    is_feature_key,
)
from hyped.utils.feature_checks import (
    get_sequence_length,
    raise_feature_is_sequence,
)


@dataclass
class ChunkSequenceConfig(BaseDataProcessorConfig):
    t: Literal[
        "hyped.data.processors.sequence.chunk"
    ] = "hyped.data.processors.sequence.chunk"

    sequence: FeatureKey | list[FeatureKey] = None
    chunk_size: int = None
    chunk_stride: int = None
    drop_last: bool = False

    def __post_init__(self) -> None:
        if is_feature_key(self.sequence):
            # make sure sequence is a list of feature keys
            self.sequence = [self.sequence]

        if self.chunk_stride is None:
            # set default value of chunk stride
            self.chunk_stride = self.chunk_size


class ChunkSequence(BaseDataProcessor[ChunkSequenceConfig]):
    @property
    def in_feature_sequence_length(self) -> int:
        return get_sequence_length(
            get_feature_at_key(self.in_features, self.config.sequence[0])
        )

    def map_features(self, features: Features) -> Features:
        chunk_features = {}
        # collect the sequence features to chunk
        for k in self.config.sequence:
            # check key type
            if not isinstance(k, str):
                raise NotImplementedError(
                    "Chunk Sequence Processor currently only supports "
                    "simple string-like feature keys, got %s" % str(k)
                )
            # make sure the feature is a sequence
            f = get_feature_at_key(features, k)
            raise_feature_is_sequence(k, f)
            # add it to the collection
            chunk_features[k] = f

        # compute all sequence lengths
        lengths = list(map(get_sequence_length, chunk_features.values()))
        # make sure they match
        if any(lengths[0] != length for length in lengths[1:]):
            raise TypeError(
                "Cannot chunk along the given sequences: "
                "Lengths of the sequences mismatch, got %s" % lengths
            )

        # length match
        length = lengths[0]

        if self.config.drop_last:
            # chunks that have smaller sizes are discarded
            out_length = self.config.chunk_size

        elif (length > 0) and (
            length - self.config.chunk_size
        ) % self.config.chunk_stride == 0:
            # the sequences can be chunked perfectly and
            # each chunk has the exact chunk size
            out_length = self.config.chunk_size

        else:
            # the last chunk might have smaller size
            out_length = -1

        # set output lengths in collected features
        for f in chunk_features.values():
            f.length = out_length

        return chunk_features

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # collect all sequences to chunk over
        sequences = {
            k: get_value_at_key(example, k) for k in self.config.sequence
        }

        if self.in_feature_sequence_length == -1:
            # check if the sequence lengths of all sequences
            # match up in case they are not fixed by the features
            lengths = list(map(len, sequences.values()))
            if any(lengths[0] != length for length in lengths[1:]):
                raise TypeError(
                    "Cannot chunk along the given sequences: "
                    "Lengths of the sequences mismatch, got %s" % lengths
                )

        # compute the number of chunks for the current example
        length = len(next(iter(sequences.values())))
        num_chunks = 1 + (
            (length - self.config.chunk_size) / self.config.chunk_stride
        )
        # round up or down depending on whether the last chunk
        # should be dropped
        num_chunks = (
            int(num_chunks) if self.config.drop_last else ceil(num_chunks)
        )

        for chunk_id in range(num_chunks):
            # compute chunk borders
            chunk_start = chunk_id * self.config.chunk_stride
            chunk_end = chunk_start + self.config.chunk_size
            # yield chunk
            yield {k: s[chunk_start:chunk_end] for k, s in sequences.items()}
