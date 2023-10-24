from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    LayoutXLMTokenizer,
)
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from datasets import Features, Sequence, Value
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class HuggingFaceTokenizerConfig(BaseDataProcessorConfig):
    """HuggingFace (Transformers) Tokenizer Config

    Data Processor applying a pre-trained huggingface tokenizer.
    For more information please refer to the documentation of
    the hugginface transformers `PreTrainedTokenizer` class.

    Type Identifier: `"hyped.data.processors.tokenizer.hf'

    Attributes:
        tokenizer (str | PreTrainedTokenizer):
            the pre-trained tokenizer to apply. Either the uri or local
            path to load the tokenizer from using `AutoTokenizer`, or
            the pre-trained tokenizer object.
        text (None | str):
            feature column to pass as `text` keyword argument
            to the tokenizer. Defaults to `text`.
        text_pair (None | str):
            feature column to pass as `text_pair` keyword argument
            to the tokenizer
        text_target (None | str):
            feature column to pass as `text_target` keyword argument
            to the tokenizer
        text_pair_target (None | str):
            feature column to pass as `text_pair_target` keyword argument
            to the tokenizer
        boxes (None | str):
            feature column to pass as `boxes` to the tokenizer.
            Required for `LayoutXLMTokenizer`.
        add_special_tokens (None | bool):
            whether to add special tokens to the tokenized sequence.
            Defaults to true.
        padding (bool | str | PaddingStrategy):
            Activate and control padding of the tokenized sequence.
            Defaults to False. See `PreTrainedTokenizer.call` for more
            information.
        truncation (bool | str | TruncationSrategy):
            Activate and control truncation of the tokenized sequence.
            Defaults to False. See `PreTrainedTokenizer.call` for more
            information.
        max_length (None | str):
            the maximum length to be used by the truncation and padding
            strategy.
        stride (int):
            overflowing tokens will have overlap of this value with the
            truncated sequence. Defaults to zero, effectively deactivating
            striding.
        is_split_into_words (bool):
            whether the text inputs are already pre-tokenized into words.
            Defaults to false.
        pad_to_multiple_of (None | int):
            when set, the sequence will be padded to a multiple of
            the specified value
        return_token_type_ids (bool):
            whether to return the token type ids
        return_attention_mask (bool):
            whether to return the attention mask
        return_overflowing_tokens (bool):
            whether to return the overflowing tokens
        return_special_tokens_mask (bool):
            whether to return the special tokens mask
        return_offsets_mapping (bool):
            whether to return the character offset mapping. Only availabel
            for fast tokenizers.
        return_length (bool):
            whether to return the sequence length
        return_word_ids (bool):
            whether to return the word ids for each token. Only available
            for fast tokenizers. Defaults to false.
    """

    t: Literal[
        "hyped.data.processors.tokenizer.hf"
    ] = "hyped.data.processors.tokenizer.hf"

    tokenizer: str | PreTrainedTokenizer = "bert-base-uncased"
    # text input to tokenize
    text: None | str = "text"
    text_pair: None | str = None
    text_target: None | str = None
    text_pair_target: None | str = None
    boxes: None | str = None
    # post-processing
    add_special_tokens: bool = True
    padding: bool | str | PaddingStrategy = False
    truncation: bool | str | TruncationStrategy = False
    max_length: None | int = None
    stride: int = 0
    is_split_into_words: bool = False
    pad_to_multiple_of: None | int = None
    # output features
    return_token_type_ids: bool = False
    return_attention_mask: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    return_word_ids: bool = False


class HuggingFaceTokenizer(BaseDataProcessor[HuggingFaceTokenizerConfig]):
    """HuggingFace (Transformers) Tokenizer

    Data Processor applying a pre-trained huggingface tokenizer.
    For more information please refer to the documentation of
    the hugginface transformers `PreTrainedTokenizer` class.
    """

    # feature keys for which the value must be extracted from the example
    # and who's values are specified directly in the configuration
    KWARGS_FROM_EXAMPLE = [
        "text",
        "text_pair",
        "text_target",
        "text_pair_target",
        "boxes",
    ]
    KWARGS_FROM_CONFIG = [
        "add_special_tokens",
        "padding",
        "truncation",
        "max_length",
        "stride",
        "is_split_into_words",
        "pad_to_multiple_of",
        "return_token_type_ids",
        "return_attention_mask",
        "return_special_tokens_mask",
        "return_offsets_mapping",
        "return_length",
    ]

    def __init__(self, config: HuggingFaceTokenizerConfig) -> None:
        super(HuggingFaceTokenizer, self).__init__(config)
        # prepare tokenizer
        tokenizer = self.config.tokenizer
        tokenizer = (
            tokenizer
            if isinstance(
                tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
            )
            else AutoTokenizer.from_pretrained(
                tokenizer, use_fast=True, add_prefix_space=True
            )
        )
        self.tokenizer = tokenizer

        # check if requested functionality is present
        if not tokenizer.is_fast:
            if self.config.return_offsets_mapping:
                raise ValueError(
                    "Offsets mapping is only available for fast "
                    "tokenizers, got %s" % self.tokenizer
                )
            if self.config.return_word_ids:
                raise ValueError(
                    "Word IDs is only available for fast tokenizers"
                    ",got %s" % self.tokenizer
                )

    def _check_text_feature(self, key: str, features: Features) -> None:
        """Check textual input dataset features

        Raises KeyError when the key is not present in the feature mapping.
        Raises TypeError when the feature type is invalid, i.e. when the
        feature is of type string but expected token sequence or vise versa.

        Arguments:
            key (str): feature key/name to check
            features (Features): feature mapping
        """
        # make sure text column is present
        if key not in features:
            raise KeyError("`%s` not present in features!" % self.config.text)

        # check type of input feature
        f = features[key]
        if self.config.is_split_into_words and not (
            isinstance(f, Sequence) and (f.feature == Value("string"))
        ):
            raise TypeError(
                "Input feature `%s` must be sequence of strings, got %s."
                % (key, f)
            )
        elif (not self.config.is_split_into_words) and (f != Value("string")):
            raise TypeError(
                "Input feature `%s` must be string, got %s." % (key, f)
            )

    def _check_input_features(self, features: Features) -> None:
        """Check input features"""
        # make sure some input is specified
        if self.config.text is None:
            raise ValueError("No text input to tokenizer specified")

        # check text input features
        for key in ["text", "text_pair", "text_target", "text_pair_target"]:
            if getattr(self.config, key) is not None:
                self._check_text_feature(getattr(self.config, key), features)

        # special case for layout-xlm
        if isinstance(self.tokenizer, LayoutXLMTokenizer):
            if not self.config.is_split_into_words:
                raise ValueError(
                    "`LayoutXLMTokenizer` expects pre-tokenized inputs"
                )

            if self.config.boxes is None:
                raise ValueError(
                    "`LayoutXLMTokenizer` requires boxes argument containing "
                    "word-level bounding boxes"
                )

            if self.config.boxes not in features:
                raise KeyError(
                    "`%s` not present in features!" % self.config.boxes
                )

            f = features[self.config.boxes]
            if not isinstance(f, Sequence) or (
                f.feature != Sequence(Value["int32"], length=4)
            ):
                raise TypeError(
                    "Expected feature type of `%s` to be sequence of "
                    "sequences of four integers, got %s"
                    % (self.config.boxes, f)
                )

    def _get_output_sequence_length(self) -> int:
        """Infers the (fixed) sequence length of the output sequences such
        as the `input_ids` and `attention_mask` given the config. Returns
        -1 when the sequence length is not guaranteed to be constant.

        Returns:
            length (int): the sequence length of output sequences
        """
        # check for constant length
        is_constant = (
            (self.config.max_length is not None)
            and (self.config.padding == "max_length")
            and (
                self.config.truncation
                in (True, "longest_first", "only_first", "only_second")
            )
        )
        # get sequence length in case it's constant
        return self.config.max_length if is_constant else -1

    def _build_output_features(self) -> Features:
        """Build the output feature mapping based on the return options
        set in the config

        Returns:
            features (Features): output feature mapping
        """
        # infer the output sequence length given the configuration
        length = self._get_output_sequence_length()

        out_features = Features()
        # add all fixed-length integer sequence outputs to features
        for key in [
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "special_tokens_mask",
            "word_ids",
        ]:
            if (key == "input_ids") or getattr(self.config, "return_%s" % key):
                out_features[key] = Sequence(
                    Value(dtype="int32"), length=length
                )

        if self.config.return_offsets_mapping:
            out_features["offset_mapping"] = Sequence(
                Sequence(Value("int32"), length=2), length=length
            )

        if self.config.return_length:
            # length output is nested into a sequence of length one
            out_features["length"] = Sequence(Value("int32"), length=1)

        return out_features

    def map_features(self, features: Features) -> Features:
        # check features and build output features
        self._check_input_features(features)
        return self._build_output_features()

    def _build_kwargs(self, example: dict[str, Any]) -> dict[str, Any]:
        kwargs = {}
        # collect all features form the example
        for key in type(self).KWARGS_FROM_EXAMPLE:
            if getattr(self.config, key) is not None:
                kwargs[key] = example[getattr(self.config, key)]
        # add all options kwargs specified in the config
        for key in type(self).KWARGS_FROM_CONFIG:
            kwargs[key] = getattr(self.config, key)
        # return the keyword arguments
        return kwargs

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # apply tokenizer
        kwargs = self._build_kwargs(example)
        enc = self.tokenizer(**kwargs)
        # add word ids to output
        if self.config.return_word_ids:
            word_ids = enc.word_ids()
            word_ids = [(i if i is not None else -1) for i in word_ids]
            enc["word_ids"] = word_ids
        # convert to dict and return
        return dict(enc)
