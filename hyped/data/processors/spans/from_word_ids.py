from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_feature_exists,
    raise_feature_is_sequence,
)
from datasets import Features, Sequence, Value
from itertools import groupby
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class TokenSpansFromWordIdsConfig(BaseDataProcessorConfig):
    """Token Spans from Word Ids Processor Config

    Convert word-ids to token-level spans. Word-ids are typically
    provided by the tokenizer (see `HuggingFaceTokenizer`).

    Attributes:
        word_ids (str):
            column containing the word-ids to parse. Defaults to `word_ids`
    """

    t: Literal[
        "hyped.data.processors.spans.from_word_ids"
    ] = "hyped.data.processors.spans.from_word_ids"

    word_ids: str = "word_ids"


class TokenSpansFromWordIds(BaseDataProcessor[TokenSpansFromWordIdsConfig]):
    """Token Spans from Word Ids Processor Config

    Convert word-ids to token-level spans. Word-ids are typically
    provided by the tokenizer (see `HuggingFaceTokenizer`).
    """

    def __init__(
        self,
        config: TokenSpansFromWordIdsConfig = TokenSpansFromWordIdsConfig(),
    ) -> None:
        super(TokenSpansFromWordIds, self).__init__(config)

    def map_features(self, features: Features) -> Features:
        """Check word-ids feature and return token level span features

        Arguments:
            features (Features): input features

        Returns:
            out (Features): span features
        """
        # make sure word ids feature exists and is a sequence of indices
        raise_feature_exists(self.config.word_ids, features)
        raise_feature_is_sequence(
            self.config.word_ids, features[self.config.word_ids], INDEX_TYPES
        )
        # return token-level span features
        return {
            "token_spans_begin": Sequence(Value("int32")),
            "token_spans_end": Sequence(Value("int32")),
        }

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Apply processor to an example

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): token-level spans
        """
        word_ids = example[self.config.word_ids]
        # check word ids
        for k, (i, j) in enumerate(zip(word_ids[:-1], word_ids[1:])):
            # must be monotonically increasing
            if j < i:
                raise ValueError(
                    "Word id sequence is invalid, got %s"
                    % word_ids[max(0, k - 4) : k + 4]  # noqa: E203
                )

        # group tokens by the word they are a part of
        # note that the word ids are sorted thus we don't
        # need to sort them explicitely before groupby
        word_groups = groupby(range(len(word_ids)), key=word_ids.__getitem__)

        spans_begin, spans_end = [], []
        # build span features from word groups
        for _, group in word_groups:
            group = tuple(group)
            assert len(group) > 0
            # token spans are exclusive, thus + 1
            spans_begin.append(min(group))
            spans_end.append(max(group) + 1)

        # return span features
        return {"token_spans_begin": spans_begin, "token_spans_end": spans_end}
