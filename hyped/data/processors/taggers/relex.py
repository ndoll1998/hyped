import numpy as np
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_feature_exists,
    raise_feature_equals,
    raise_feature_is_sequence,
    get_sequence_length,
)
from hyped.utils.spans import make_spans_exclusive
from dataclasses import dataclass
from datasets import Features, Sequence, Value
from typing import Any, Literal


@dataclass
class RelExTaggerConfig(BaseDataProcessorConfig):
    """Relation Extraction Tagger

    Marks source and target entities in the input sequence.

    Attributes:
        source_begin_marker (str | int):
            marker used to indicate the beginning of the source entity.
            Marker type should match the item type of the input sequence,
            i.e. string for token sequence and integer for token id sequence.
        source_end_marker (str | int):
            marker used to indicate the end of the source entity.
        target_begin_marker (str | int):
            marker used to indicate the begging of the target entity.
        target_end_marker (str | int):
            marker used to indicate the end of the target entity.
        input_sequence (str):
            column containing the input sequence in which to mark
            the related entities
        source_span_begin (str):
            column containing the begin value of the source entity span wrt.
            the input sequence
        source_span_end (str):
            column containing the end value of the source entity span wrt.
            the input sequence
        target_span_begin (str):
            column containing the begin value of the target entity span wrt.
            the input sequence
        target_span_end (str):
            column containing the end value of the target entity span wrt.
            the input sequence
        source_span_inclusive (bool):
            whether the end coordinate of the source span is
            inclusive or exclusive. Defaults to false.
        target_span_inclusive (bool):
            whether the end coordinate of the target span is
            inclusive or exclusive. Defaults to false.
        max_sequence_length (None | int):
            if set the input sequence is truncated around the entities to the
            specified adhere to the specified maximum sequence length.
            Examples where the distance between entities exceeds the maximum
            length are filtered
    """

    t: Literal[
        "hyped.data.processors.taggers.relex"
    ] = "hyped.data.processors.taggers.relex"

    # source entity markers
    source_begin_marker: str | int = None
    source_end_marker: str | int = None
    # target entity markers
    target_begin_marker: str | int = None
    target_end_marker: str | int = None

    input_sequence: str = None
    # source entity span
    source_span_begin: str = None
    source_span_end: str = None
    # target entity span
    target_span_begin: str = None
    target_span_end: str = None

    # span inclusive or not
    source_span_inclusive: bool = False
    target_span_inclusive: bool = False

    # maximum allowed sequence length
    max_sequence_length: None | int = None

    @property
    def markers(self) -> list[str | int]:
        """List of Source and Target markers"""
        return [
            self.source_begin_marker,
            self.source_end_marker,
            self.target_begin_marker,
            self.target_end_marker,
        ]


class RelExTagger(BaseDataProcessor[RelExTaggerConfig]):
    """Relation Extraction Tagger

    Marks source and target entities in the input sequence.
    """

    def _marked_sequence_feature(self, features: Features) -> Sequence:
        sequence = features[self.config.input_sequence]

        # increase length by four to account for the entity markers
        length = get_sequence_length(sequence)
        length = -1 if length == -1 else (length + 4)
        # apply maximum sequence length
        if self.config.max_sequence_length is not None:
            length = min(length, self.config.max_sequence_length)

        return Sequence(sequence.feature, length=length)

    def _get_sequence_value_type(self) -> Value | list[Value]:
        """Check the marker configuration and infer the expected
        value type of the items in the input sequence from it

        Returns:
            value_type (Value | list[Value]): expected value type(s)
        """

        if (
            isinstance(self.config.source_begin_marker, str)
            and isinstance(self.config.source_end_marker, str)
            and isinstance(self.config.target_begin_marker, str)
            and isinstance(self.config.target_end_marker, str)
        ):
            # when the markers are of type string then the input
            # sequence is expected to be a token sequence, i.e.
            # a sequence of strings
            return Value("string")

        elif (
            isinstance(self.config.source_begin_marker, int)
            and isinstance(self.config.source_end_marker, int)
            and isinstance(self.config.target_begin_marker, int)
            and isinstance(self.config.target_end_marker, int)
        ):
            # when the markers are of type int then the input
            # sequence is expected to be a token id sequence
            return INDEX_TYPES

        else:
            # the marker types are either invalid or do not match
            raise TypeError(
                "Marker types must all either be str or int, "
                "got %s, %s, %s, %s"
                % (
                    self.config.source_begin_marker,
                    self.config.source_end_marker,
                    self.config.target_begin_marker,
                    self.config.target_end_marker,
                )
            )

    def map_features(self, features: Features) -> Features:
        # make sure the maximum sequence length value is valid if set
        if (self.config.max_sequence_length is not None) and (
            self.config.max_sequence_length <= 0
        ):
            raise ValueError(
                "Expected maximum sequence length to be a positive non-zero "
                "value, got %s" % self.config.max_sequence_length
            )

        # infer expected sequence value type from config
        value_type = self._get_sequence_value_type()

        # make sure input feature exists
        raise_feature_exists(self.config.input_sequence, features)
        raise_feature_is_sequence(
            self.config.input_sequence,
            features[self.config.input_sequence],
            value_type,
        )

        for key in [
            self.config.source_span_begin,
            self.config.source_span_end,
            self.config.target_span_begin,
            self.config.target_span_end,
        ]:
            # make sure span exists and is of expected type
            raise_feature_exists(key, features)
            raise_feature_equals(key, features[key], INDEX_TYPES)

        return {
            "%s.with_markers"
            % self.config.input_sequence: self._marked_sequence_feature(
                features
            ),
        }

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get input sequence from example
        input_sequence = example[self.config.input_sequence]
        l = len(input_sequence)  # noqa: E741

        # get source and target spans
        src_span = (
            example[self.config.source_span_begin],
            example[self.config.source_span_end],
        )
        tgt_span = (
            example[self.config.target_span_begin],
            example[self.config.target_span_end],
        )
        # make spans exclusive, that is the end coordinate points
        # to the first item after the entity as the marker will be
        # inserted before the item
        src_span = make_spans_exclusive(
            [src_span], self.config.source_span_inclusive
        )[0]
        tgt_span = make_spans_exclusive(
            [tgt_span], self.config.target_span_inclusive
        )[0]
        # concatenate spans for ease of use later
        spans = np.asarray([*src_span, *tgt_span], dtype=int)

        if self.config.max_sequence_length is not None:
            i, j = min(spans), max(spans)
            # check if the example exceeds the maximum sequence length
            if j - i > self.config.max_sequence_length:
                # filter out the example
                return

            # compute the budget of tokens to spend
            # accounting for the four markers
            d = self.config.max_sequence_length - 4 - (j - i)
            # compute sub-sequence span containing source
            # and target entities
            i = max(0, i - d // 2)
            j = min(l, i + self.config.max_sequence_length - 4)
            i = max(0, j - (self.config.max_sequence_length - 4))

            # get the relevant sub-sequence
            # and update the spans accordingly
            input_sequence = input_sequence[i:j]
            spans -= i

        # insert markers
        marked_input_sequence = np.insert(
            np.asarray(input_sequence, dtype=object),
            spans,
            np.asarray(self.config.markers, dtype=object),
        )

        # remove dummy item at the end of the sequence and return
        yield {
            "%s.with_markers"
            % self.config.input_sequence: marked_input_sequence.tolist()
        }
