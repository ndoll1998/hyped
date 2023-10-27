import numpy as np
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_feature_exists,
    raise_features_align,
    raise_feature_is_sequence,
    check_feature_equals,
    check_feature_is_sequence,
)
from hyped.utils.spans import compute_spans_overlap_matrix
from datasets import Features
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class CharToTokenSpansConfig(BaseDataProcessorConfig):
    """Span Conversion Data Processor Config

    Convert character-level span annotations to token-level
    span annotations. Especially useful for squad-style Question
    Answering (QA) or Named-Entity-Recognition (NER).

    Note that the output spans are exclusive, i.e. for an output
    span (i, j) the index of the last member to the span is j-1.

    Type Identifier: `hyped.data.processors.helpers.char_to_token_spans`

    Attributes:
        char_spans_begin (str):
            column containing begins of character-level span annotations
        char_spans_end (str):
            column containing ends of character-level span annotations
        token_offsets_begin (str):
            column containing begins of token-offsets
        token_offsets_end (str):
            column containing ends of token-offsets
        char_spans_inclusive (bool):
            whether the end coordinates of the character spans are
            inclusive or exclusive. Defaults to false.
        token_offsets_inclusive (bool):
            whether the end coordinates of the token offsets are
            inclusive or exclusive. Defaults to false.
    """

    t: Literal[
        "hyped.data.processors.helpers.char_to_token_spans"
    ] = "hyped.data.processors.helpers.char_to_token_spans"

    # character-level annotation spans
    char_spans_begin: str = None
    char_spans_end: str = None
    # token-level offsets
    token_offsets_begin: str = None
    token_offsets_end: str = None
    # whether the end coordinates are inclusive of exclusive
    char_spans_inclusive: bool = False
    token_offsets_inclusive: bool = False


class CharToTokenSpans(BaseDataProcessor[CharToTokenSpansConfig]):
    """Span Conversion Data Processor

    Convert character-level span annotations to token-level
    span annotations. Especially useful for squad-style Question
    Answering (QA) or Named-Entity-Recognition (NER).

    Note that the output spans are exclusive, i.e. for an output
    span (i, j) the index of the last member to the span is j-1.
    """

    def map_features(self, features: Features) -> Features:
        """Check input features and return feature mapping
        for token-level span annotations.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): token-level span annotation features
        """
        # make sure all features exisr
        raise_feature_exists(self.config.char_spans_begin, features)
        raise_feature_exists(self.config.char_spans_end, features)
        raise_feature_exists(self.config.token_offsets_begin, features)
        raise_feature_exists(self.config.token_offsets_end, features)

        # character spans must either be a sequence of
        # integers or an integer value
        for key in [self.config.char_spans_begin, self.config.char_spans_end]:
            if not (
                check_feature_is_sequence(features[key], INDEX_TYPES)
                or check_feature_equals(features[key], INDEX_TYPES)
            ):
                raise TypeError(
                    "Expected `%s` to be an integer value or a sequence "
                    "of integers, got %s" % (key, features[key])
                )

        # character spans begin and end features must align
        raise_features_align(
            self.config.char_spans_begin,
            self.config.char_spans_end,
            features[self.config.char_spans_begin],
            features[self.config.char_spans_end],
        )

        # token offsets must be sequence of integers
        raise_feature_is_sequence(
            self.config.token_offsets_begin,
            features[self.config.token_offsets_begin],
            INDEX_TYPES,
        )
        raise_feature_is_sequence(
            self.config.token_offsets_begin,
            features[self.config.token_offsets_end],
            INDEX_TYPES,
        )
        # and they must align excatly
        raise_features_align(
            self.config.token_offsets_begin,
            self.config.token_offsets_end,
            features[self.config.token_offsets_begin],
            features[self.config.token_offsets_end],
        )

        return Features(
            {
                "token_spans_begin": features[self.config.char_spans_begin],
                "token_spans_end": features[self.config.char_spans_end],
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
            out (dict[str, Any]): token-level span annotations
        """
        # get character level spans
        char_spans_begin = example[self.config.char_spans_begin]
        char_spans_end = example[self.config.char_spans_end]
        # check if it is a single value
        is_value = isinstance(char_spans_begin, int)
        # pack value into list
        if is_value:
            char_spans_begin = [char_spans_begin]
            char_spans_end = [char_spans_end]

        # get spans and offsets
        char_spans = zip(char_spans_begin, char_spans_end)
        token_offsets = zip(
            example[self.config.token_offsets_begin],
            example[self.config.token_offsets_end],
        )

        # for each character-level span find the
        # token offset that it overlaps with
        mask = compute_spans_overlap_matrix(
            char_spans,
            token_offsets,
            self.config.char_spans_inclusive,
            self.config.token_offsets_inclusive,
        )

        # get begins and ends from mask
        token_spans_begin = mask.argmax(axis=1)
        token_spans_end = token_spans_begin + mask.sum(axis=1)
        # convert to list
        token_spans_begin = token_spans_begin.tolist()
        token_spans_end = token_spans_end.tolist()

        # return value when input was also a single value
        if is_value:
            token_spans_begin = token_spans_begin[0]
            token_spans_end = token_spans_end[0]

        # return token-level spans
        return {
            "token_spans_begin": token_spans_begin,
            "token_spans_end": token_spans_end,
        }
