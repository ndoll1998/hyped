import numpy as np
from .outputs import SpansOutputs
from hyped.data.processors.tokenizers.hf import HuggingFaceTokenizerOutputs
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_feature_is_sequence,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from datasets import Features, Sequence, Value
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class TokenSpansFromWordIdsConfig(BaseDataProcessorConfig):
    """Token Spans from Word Ids Processor Config

    Convert word-ids to token-level spans. Word-ids are typically
    provided by the tokenizer (see `HuggingFaceTokenizer`).

    Attributes:
        word_ids (FeatureKey):
            column containing the word-ids to parse.
            Defaults to `HuggingFaceTokenizerOutputs.WORD_IDS`
        mask (None | FeatureKey):
            column containing mask over words indicating which items
            in the `word_ids` sequence to ignore for building the spans.
            If set to None, all values in the word-ids sequence are
            considered. Defaults to
            `HuggingFaceTokenizerOutputs.SPECIAL_TOKENS_MASK`.
    """

    t: Literal[
        "hyped.data.processors.spans.from_word_ids"
    ] = "hyped.data.processors.spans.from_word_ids"

    word_ids: FeatureKey = HuggingFaceTokenizerOutputs.WORD_IDS
    mask: None | FeatureKey = HuggingFaceTokenizerOutputs.SPECIAL_TOKENS_MASK


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
        word_ids = get_feature_at_key(features, self.config.word_ids)
        raise_feature_is_sequence(self.config.word_ids, word_ids, INDEX_TYPES)
        # make sure mask is valid if specified
        if self.config.mask is not None:
            mask = get_feature_at_key(features, self.config.mask)
            raise_feature_is_sequence(
                self.config.mask,
                mask,
                [Value("bool")] + INDEX_TYPES,
            )
        # return token-level span features
        return {
            SpansOutputs.BEGINS: Sequence(Value("int32")),
            SpansOutputs.ENDS: Sequence(Value("int32")),
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
        # get word ids from example and convert to numpy array
        word_ids = get_value_at_key(example, self.config.word_ids)
        word_ids = np.asarray(word_ids)

        if self.config.mask is not None:
            # get mask from example and convert to numpy array
            # also invert it to get a mask indicating valid items
            mask = get_value_at_key(example, self.config.mask)
            mask = ~np.asarray(mask).astype(bool)
        else:
            # create a dummy mask of all trues when mask is not specified
            mask = np.full_like(word_ids, fill_value=True, dtype=bool)

        # apply mask to word ids
        masked_word_ids = word_ids[mask]

        # check word ids
        if (masked_word_ids[:-1] > masked_word_ids[1:]).any():
            raise ValueError(
                "Word id sequence must be monotonically increasing, got %s"
                % masked_word_ids
            )

        word_bounds_mask = word_ids[:-1] != word_ids[1:]
        # identify the beginnings of words
        word_begins_mask = np.append(True, word_bounds_mask)
        word_begins_mask &= mask
        # identify the ends of words
        word_ends_mask = np.append(word_bounds_mask, True)
        word_ends_mask &= mask
        # get the indices, that is the index spans
        (word_begins,) = word_begins_mask.nonzero()
        (word_ends,) = word_ends_mask.nonzero()
        # make word-spans exclusive
        word_ends += 1

        return {
            SpansOutputs.BEGINS: word_begins.tolist(),
            SpansOutputs.ENDS: word_ends.tolist(),
        }
