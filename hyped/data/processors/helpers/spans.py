import numpy as np
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from datasets import Features, Sequence, Value
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class CharToTokenSpansConfig(BaseDataProcessorConfig):
    """Span Conversion Data Processor Config

    Convert character-level span annotations to token-level
    span annotations. Especially useful for squad-style Question
    Answering (QA) or Named-Entity-Recognition (NER).

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
    """

    def map_features(self, features: Features) -> Features:
        """Check input features and return feature mapping
        for token-level span annotations.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): token-level span annotation features
        """
        if self.config.char_spans_begin not in features:
            raise KeyError(
                "`%s` not present in features!" % self.config.char_spans_begin
            )

        if self.config.char_spans_end not in features:
            raise KeyError(
                "`%s` not present in features!" % self.config.char_spans_end
            )

        if self.config.token_offsets_begin not in features:
            raise KeyError(
                "`%s` not present in features!"
                % self.config.token_offsets_begin
            )

        if self.config.token_offsets_end not in features:
            raise KeyError(
                "`%s` not present in features!" % self.config.token_offsets_end
            )

        # check character span feature
        # must be either Sequence of integers or an integer value
        if not (
            isinstance(features[self.config.char_spans_begin], Sequence)
            and features[self.config.char_spans_begin].feature
            == Value("int32")
        ) and not (features[self.config.char_spans_begin] == Value("int32")):
            raise TypeError(
                "Expected `char_spans_begin` to be an integer value "
                "or a sequence of integers, got %s"
                % features[self.config.char_spans_begin]
            )

        if not (
            isinstance(features[self.config.char_spans_end], Sequence)
            and features[self.config.char_spans_end].feature == Value("int32")
        ) and not (features[self.config.char_spans_end] == Value("int32")):
            raise TypeError(
                "Expected `char_spans_begin` to be an integer value "
                "or a sequence of integers, got %s"
                % features[self.config.char_spans_end]
            )

        if (
            features[self.config.char_spans_begin]
            != features[self.config.char_spans_end]
        ):
            raise TypeError(
                "Begin and end features of character spans don't match, "
                "got %s != %s"
                % (
                    features[self.config.char_spans_begin],
                    features[self.config.char_spans_end],
                )
            )

        # check token offset features
        if not (
            isinstance(features[self.config.token_offsets_begin], Sequence)
            and features[self.config.token_offsets_begin].feature
            == Value("int32")
        ):
            raise TypeError(
                "Expected `token_offsets_begin` to be a sequence of integers, "
                "got %s" % features[self.config.token_offsets_begin]
            )

        if not (
            isinstance(features[self.config.token_offsets_end], Sequence)
            and features[self.config.token_offsets_end].feature
            == Value("int32")
        ):
            raise TypeError(
                "Expected `token_offsets_begin` to be a sequence of integers, "
                "got %s" % features[self.config.token_offsets_end]
            )

        if (
            features[self.config.token_offsets_begin]
            != features[self.config.token_offsets_end]
        ):
            raise TypeError(
                "Begin and end features of token offsets don't match, "
                "got %s != %s"
                % (
                    features[self.config.token_offsets_begin],
                    features[self.config.token_offsets_end],
                )
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

        # get spans and offsets and convert to numpy arrays
        char_spans_begin = np.asarray(char_spans_begin)
        char_spans_end = np.asarray(char_spans_end)
        token_offsets_begin = np.asarray(
            example[self.config.token_offsets_begin]
        )
        token_offsets_end = np.asarray(example[self.config.token_offsets_end])
        # make end coordinates exclusive
        char_spans_end += int(self.config.char_spans_inclusive)
        token_offsets_end += int(self.config.token_offsets_inclusive)
        # compute mask over token spans
        mask = (char_spans_begin[:, None] <= token_offsets_begin[None, :]) & (
            token_offsets_end[None, :] < char_spans_end[:, None]
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
