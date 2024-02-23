from collections import defaultdict
from functools import partial

import pytest
from datasets import Features, Sequence, Value
from transformers import AutoTokenizer

from hyped.data.processors.tokenizers.hf import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


@pytest.fixture(params=["bert-base-uncased", "bert-base-german-cased"])
def tokenizer(request):
    return request.param


class TestHuggingFaceTokenizer(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features({"text": Value("string")})

    @pytest.fixture
    def in_batch(self):
        return {
            "text": [
                "Apple Inc. is expected to announce a new product at the "
                "upcoming conference in San Francisco.",
                "The United States President, Joe Biden, addressed the "
                "nation on climate change and economic policies.",
                "Scientists at NASA are conducting experiments to explore "
                "the possibility of life on Mars.",
                "The film, directed by Christopher Nolan, received critical "
                "acclaim and won several awards.",
                "Researchers from Oxford University published a "
                "groundbreaking study on artificial intelligence last month.",
            ]
        }

    @pytest.fixture(params=[-1, 8, 32, 128])
    def max_length(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [],
            ["return_tokens"],
            ["return_token_type_ids"],
            ["return_attention_mask"],
            ["return_special_tokens_mask"],
            ["return_offsets_mapping"],
            ["return_length"],
            ["return_word_ids"],
            [
                "return_tokens",
                "return_token_type_ids",
                "return_attention_mask",
                "return_special_tokens_mask",
                "return_offsets_mapping",
                "return_length",
                "return_word_ids",
            ],
        ]
    )
    def return_options(self, request):
        return {option: True for option in request.param}

    @pytest.fixture
    def expected_out_batch(self, processor, in_batch):
        config = processor.config
        tokenize = partial(
            # use the tokenizer of the processor
            processor.tokenizer,
            # specify all keyword arguments as set in the config
            add_special_tokens=config.add_special_tokens,
            padding=config.padding,
            truncation=config.truncation,
            max_length=config.max_length,
            stride=config.stride,
            is_split_into_words=config.is_split_into_words,
            pad_to_multiple_of=config.pad_to_multiple_of,
            return_token_type_ids=config.return_token_type_ids,
            return_attention_mask=config.return_attention_mask,
            return_special_tokens_mask=config.return_special_tokens_mask,
            return_offsets_mapping=config.return_offsets_mapping,
            return_length=config.return_length,
        )
        # apply to each text
        out_batch = defaultdict(list)
        for enc in map(tokenize, in_batch["text"]):
            # add to output batch
            for key, val in enc.items():
                # length is a list of a single item
                # when only a single element is passed
                if key == "length":
                    val = val[0]
                # offset mapping items are typically returned
                # as tuples, however the processor returns
                # a list of lists
                if key == "offset_mapping":
                    val = list(map(list, val))
                # collect features
                out_batch[key].append(val)
            # add tokens
            if processor.config.return_tokens:
                out_batch["tokens"].append(
                    processor.tokenizer.convert_ids_to_tokens(enc.input_ids)
                )
            # add word ids
            if processor.config.return_word_ids:
                word_ids = enc.word_ids()
                word_ids = [(i if i is not None else -1) for i in word_ids]
                out_batch["word_ids"].append(word_ids)

        return dict(out_batch)

    @pytest.fixture
    def processor(self, tokenizer, max_length, return_options):
        if max_length > 0:
            return HuggingFaceTokenizer(
                HuggingFaceTokenizerConfig(
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    tokenizer=tokenizer,
                    **return_options,
                )
            )
        else:
            return HuggingFaceTokenizer(
                HuggingFaceTokenizerConfig(
                    tokenizer=tokenizer, **return_options
                )
            )

    @pytest.fixture
    def expected_out_features(self, max_length, return_options):
        features = Features(
            {"input_ids": Sequence(Value("int32"), length=max_length)}
        )

        if return_options.get("return_tokens", False):
            features["tokens"] = Sequence(Value("string"), length=max_length)
        if return_options.get("return_token_type_ids", False):
            features["token_type_ids"] = Sequence(
                Value("int32"), length=max_length
            )
        if return_options.get("return_attention_mask", False):
            features["attention_mask"] = Sequence(
                Value("int32"), length=max_length
            )
        if return_options.get("return_special_tokens_mask", False):
            features["special_tokens_mask"] = Sequence(
                Value("int32"), length=max_length
            )
        if return_options.get("return_offsets_mapping", False):
            features["offset_mapping"] = Sequence(
                Sequence(Value("int32"), length=2), length=max_length
            )
        if return_options.get("return_length", False):
            features["length"] = Value("int32")
        if return_options.get("return_word_ids", False):
            features["word_ids"] = Sequence(Value("int32"), length=max_length)

        return features


class TestHuggingFaceTokenizerPreparationErrors(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                # text inputs
                "text": Value("string"),
                "text_pair": Value("string"),
                "text_target": Value("string"),
                "text_pair_target": Value("string"),
                # pre-tokenized text inputs
                "tok_text": Sequence(Value("string")),
                "tok_text_pair": Sequence(Value("string")),
                "tok_text_target": Sequence(Value("string")),
                "tok_text_pair_target": Sequence(Value("string")),
            }
        )

    @pytest.fixture(params=[False, True])
    def is_split_into_words(self, request):
        return request.param

    @pytest.fixture(
        params=[
            ["text"],
            ["text", "text_pair"],
            ["text", "text_pair", "text_target"],
            ["text", "text_pair", "text_target", "text_pair_target"],
        ]
    )
    def text_features(self, request):
        return request.param

    @pytest.fixture
    def processor(
        self, text_features, is_split_into_words, expected_err_on_prepare
    ):
        # build valid keyword arguments
        kwargs = dict(
            zip(
                text_features,
                map(
                    lambda k: ("tok_%s" if is_split_into_words else "%s") % k,
                    text_features,
                ),
            )
        )
        # add invalid key to kwargs
        if expected_err_on_prepare is KeyError:
            kwargs[text_features[-1]] = "INVALID_KEY"
        # change to invalid input type
        if expected_err_on_prepare is TypeError:
            key = text_features[-1]
            kwargs[key] = ("%s" if is_split_into_words else "tok_%s") % key
        # create tokenizer processor
        return HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                is_split_into_words=is_split_into_words, **kwargs
            )
        )

    @pytest.fixture(params=[None, KeyError, TypeError])
    def expected_err_on_prepare(self, request):
        return request.param


class TestHuggingFaceTokenizerErrorOnSlowTokenizer:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"return_offsets_mapping": True},
            {"return_word_ids": True},
            {"return_offsets_mapping": True, "return_word_ids": True},
            {
                "return_token_type_ids": True,
                "return_attention_mask": True,
                "return_special_tokens_mask": True,
                "return_offsets_mapping": True,
                "return_length": True,
                "return_word_ids": True,
            },
        ],
    )
    def test_error_on_slow_tokenizers(self, tokenizer, kwargs):
        # create data processor using fast tokenizer
        # this shouldn't raise an error
        HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                tokenizer=AutoTokenizer.from_pretrained(
                    tokenizer, use_fast=True
                ),
                **kwargs,
            )
        )

        # get context manager depending on the error type
        with pytest.raises(ValueError):
            # this should raise an error of the err_type if its not None
            HuggingFaceTokenizer(
                HuggingFaceTokenizerConfig(
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, use_fast=False
                    ),
                    **kwargs,
                )
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"return_token_type_ids": True},
            {"return_attention_mask": True},
            {"return_special_tokens_mask": True},
            {"return_length": True},
        ],
    )
    def test_valid_on_slow_tokenizer(self, tokenizer, kwargs):
        # create data processor using fast tokenizer
        HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                tokenizer=AutoTokenizer.from_pretrained(
                    tokenizer, use_fast=True
                ),
                **kwargs,
            )
        )
        # create data processor using slow tokenizer
        HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                tokenizer=AutoTokenizer.from_pretrained(
                    tokenizer, use_fast=False
                ),
                **kwargs,
            )
        )
