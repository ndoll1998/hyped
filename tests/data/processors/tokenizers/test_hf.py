from hyped.data.processors.tokenizers.hf import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerConfig,
)
from transformers import AutoTokenizer
from datasets import Features, Sequence, Value
from contextlib import nullcontext
import pyarrow as pa
import pytest


class TestHuggingFaceTokenizer:
    @pytest.fixture
    def features(self):
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

    @pytest.mark.parametrize(
        "err_type,config",
        [
            # no input text specified
            [ValueError, HuggingFaceTokenizerConfig(text=None)],
            # error on invalid key
            [KeyError, HuggingFaceTokenizerConfig(text="INVALID_KEY")],
            [
                KeyError,
                HuggingFaceTokenizerConfig(
                    text="text", text_pair="INVALID_KEY"
                ),
            ],
            [
                KeyError,
                HuggingFaceTokenizerConfig(
                    text="text",
                    text_pair="text_pair",
                    text_target="INVALID_KEY",
                ),
            ],
            [
                KeyError,
                HuggingFaceTokenizerConfig(
                    text="text",
                    text_pair="text_pair",
                    text_target="text_target",
                    text_pair_target="INVALID_KEY",
                ),
            ],
            # error on string input but expected pre-tokenized text
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="text", is_split_into_words=True
                ),
            ],
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="tok_text",
                    text_pair="text_pair",
                    is_split_into_words=True,
                ),
            ],
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="tok_text",
                    text_pair="tok_text_pair",
                    text_target="text_target",
                    is_split_into_words=True,
                ),
            ],
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="tok_text",
                    text_pair="tok_text_pair",
                    text_target="tok_text_target",
                    text_pair_target="text_pair_target",
                    is_split_into_words=True,
                ),
            ],
            # error on pre-tokenized input but expected string
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="tok_text", is_split_into_words=False
                ),
            ],
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="text",
                    text_pair="tok_text_pair",
                    is_split_into_words=False,
                ),
            ],
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="text",
                    text_pair="text_pair",
                    text_target="tok_text_target",
                    is_split_into_words=True,
                ),
            ],
            [
                TypeError,
                HuggingFaceTokenizerConfig(
                    text="text",
                    text_pair="text_pair",
                    text_target="text_target",
                    text_pair_target="tok_text_pair_target",
                    is_split_into_words=False,
                ),
            ],
        ],
    )
    def test_error_on_invalid_feature(self, err_type, config, features):
        with pytest.raises(err_type):
            HuggingFaceTokenizer(config).prepare(features)

    @pytest.mark.parametrize(
        "tokenizer", ["gpt2", "bert-base-uncased", "bert-base-german-cased"]
    )
    @pytest.mark.parametrize(
        "err_type,kwargs",
        [
            [None, {}],
            [None, {"return_token_type_ids": True}],
            [None, {"return_attention_mask": True}],
            [None, {"return_special_tokens_mask": True}],
            [None, {"return_length": True}],
            [ValueError, {"return_offsets_mapping": True}],
            [ValueError, {"return_word_ids": True}],
            [
                ValueError,
                {"return_offsets_mapping": True, "return_word_ids": True},
            ],
            [
                ValueError,
                {
                    "return_token_type_ids": True,
                    "return_attention_mask": True,
                    "return_special_tokens_mask": True,
                    "return_offsets_mapping": True,
                    "return_length": True,
                    "return_word_ids": True,
                },
            ],
        ],
    )
    def test_error_on_slow_tokenizers(self, tokenizer, err_type, kwargs):
        # create data processor using fast tokenizer
        # this shouldn't raise an error
        HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                tokenizer=AutoTokenizer.from_pretrained(
                    tokenizer, use_fast=True
                ),
                **kwargs
            )
        )

        # get context manager depending on the error type
        with nullcontext() if err_type is None else pytest.raises(err_type):
            # this should raise an error of the err_type if its not None
            HuggingFaceTokenizer(
                HuggingFaceTokenizerConfig(
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, use_fast=False
                    ),
                    **kwargs
                )
            )

    @pytest.mark.parametrize(
        "config,",
        [
            # text inputs
            HuggingFaceTokenizerConfig(text="text"),
            HuggingFaceTokenizerConfig(text="text", text_pair="text_pair"),
            HuggingFaceTokenizerConfig(
                text="text", text_pair="text_pair", text_target="text_target"
            ),
            HuggingFaceTokenizerConfig(
                text="text",
                text_pair="text_pair",
                text_target="text_target",
                text_pair_target="text_pair_target",
            ),
            # pre-tokenized inputs
            HuggingFaceTokenizerConfig(
                text="tok_text", is_split_into_words=True
            ),
            HuggingFaceTokenizerConfig(
                text="tok_text",
                text_pair="tok_text_pair",
                is_split_into_words=True,
            ),
            HuggingFaceTokenizerConfig(
                text="tok_text",
                text_pair="tok_text_pair",
                text_target="tok_text_target",
                is_split_into_words=True,
            ),
            HuggingFaceTokenizerConfig(
                text="tok_text",
                text_pair="tok_text_pair",
                text_target="tok_text_target",
                text_pair_target="tok_text_pair_target",
                is_split_into_words=True,
            ),
        ],
    )
    def test_valid_config(self, config, features):
        HuggingFaceTokenizer(config).prepare(features)

    @pytest.mark.parametrize(
        "truncation", [True, "longest_first", "only_first", "only_second"]
    )
    @pytest.mark.parametrize("length", [42, 128, 1337])
    def test_output_sequence_length(self, truncation, length):
        # create tokenizer processor
        t = HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                max_length=length,
                padding="max_length",
                truncation=truncation,
            )
        )
        # check max length
        assert t._get_output_sequence_length() == length

    @pytest.mark.parametrize(
        "config,out_features",
        [
            # non-fixed output sequence length
            [
                HuggingFaceTokenizerConfig(),
                Features({"input_ids": Sequence(Value("int32"))}),
            ],
            [
                HuggingFaceTokenizerConfig(
                    return_token_type_ids=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "token_type_ids": Sequence(Value("int32")),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    return_attention_mask=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "attention_mask": Sequence(Value("int32")),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    return_special_tokens_mask=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "special_tokens_mask": Sequence(Value("int32")),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(return_offsets_mapping=True),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "offsets_mapping": Sequence(
                            Sequence(Value("int32"), length=2)
                        ),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(return_length=True),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "length": Value("int32"),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(return_word_ids=True),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "word_ids": Sequence(Value("int32")),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    return_length=True,
                    return_word_ids=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "token_type_ids": Sequence(Value("int32")),
                        "attention_mask": Sequence(Value("int32")),
                        "special_tokens_mask": Sequence(Value("int32")),
                        "offsets_mapping": Sequence(
                            Sequence(Value("int32"), length=2)
                        ),
                        "length": Value("int32"),
                        "word_ids": Sequence(Value("int32")),
                    }
                ),
            ],
            # fixed output sequence length
            [
                HuggingFaceTokenizerConfig(
                    max_length=42, padding="max_length", truncation=True
                ),
                Features({"input_ids": Sequence(Value("int32"), length=42)}),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "token_type_ids": Sequence(Value("int32"), length=42),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "attention_mask": Sequence(Value("int32"), length=42),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_special_tokens_mask=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "special_tokens_mask": Sequence(
                            Value("int32"), length=42
                        ),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_offsets_mapping=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "offsets_mapping": Sequence(
                            Sequence(Value("int32"), length=2), length=42
                        ),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_length=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "length": Value("int32"),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_word_ids=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "word_ids": Sequence(Value("int32"), length=42),
                    }
                ),
            ],
            [
                HuggingFaceTokenizerConfig(
                    max_length=42,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    return_length=True,
                    return_word_ids=True,
                ),
                Features(
                    {
                        "input_ids": Sequence(Value("int32"), length=42),
                        "token_type_ids": Sequence(Value("int32"), length=42),
                        "attention_mask": Sequence(Value("int32"), length=42),
                        "special_tokens_mask": Sequence(
                            Value("int32"), length=42
                        ),
                        "offsets_mapping": Sequence(
                            Sequence(Value("int32"), length=2), length=42
                        ),
                        "length": Value("int32"),
                        "word_ids": Sequence(Value("int32"), length=42),
                    }
                ),
            ],
        ],
    )
    def test_output_features(self, config, out_features, features):
        p = HuggingFaceTokenizer(config)
        # check based on build output functionality
        assert p._build_output_features() == out_features
        # check based on prepare functionality
        p.prepare(features)
        assert p.new_features == out_features
        assert p.out_features == Features(features | out_features)

    @pytest.mark.parametrize(
        "tokenizer", ["gpt2", "bert-base-uncased", "bert-base-german-cased"]
    )
    @pytest.mark.parametrize(
        "kwargs,expected_keys",
        [
            [{}, ["input_ids"]],
            [{"return_token_type_ids": True}, ["input_ids", "token_type_ids"]],
            [{"return_attention_mask": True}, ["input_ids", "attention_mask"]],
            [
                {"return_special_tokens_mask": True},
                ["input_ids", "special_tokens_mask"],
            ],
            [
                {"return_offsets_mapping": True},
                ["input_ids", "offset_mapping"],
            ],
            [{"return_length": True}, ["input_ids", "length"]],
            [{"return_word_ids": True}, ["input_ids", "word_ids"]],
            [
                {
                    "return_token_type_ids": True,
                    "return_attention_mask": True,
                    "return_special_tokens_mask": True,
                    "return_offsets_mapping": True,
                    "return_length": True,
                    "return_word_ids": True,
                },
                [
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                    "special_tokens_mask",
                    "offset_mapping",
                    "length",
                    "word_ids",
                ],
            ],
        ],
    )
    def test_tokenize(self, tokenizer, kwargs, expected_keys):
        # text to be tokenized
        texts = [
            "Apple Inc. is expected to announce a new product at the upcoming "
            "conference in San Francisco.",
            "The United States President, Joe Biden, addressed the nation on "
            "climate change and economic policies.",
            "Scientists at NASA are conducting experiments to explore the "
            "possibility of life on Mars.",
            "The film, directed by Christopher Nolan, received critical "
            "acclaim and won several awards.",
            "Researchers from Oxford University published a groundbreaking "
            "study on artificial intelligence last month.",
        ]

        # create copies to avoid changing the original
        kwargs = kwargs.copy()
        expected_keys = expected_keys.copy()

        tokenizer_kwargs = kwargs.copy()
        tokenizer_kwargs.pop("return_word_ids", None)
        # load tokenizer and tokenize texts
        t = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        tokenized_texts = [t(text, **tokenizer_kwargs) for text in texts]

        # create tokenizer processor
        p = HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(tokenizer=t, text="text", **kwargs)
        )
        p.prepare(Features({"text": Value("string")}))
        # apply processor to tokenize texts
        batch = p.batch_process(
            {"text": texts}, index=range(len(texts)), rank=0
        )

        # check word ids in output
        if "word_ids" in expected_keys:
            assert "word_ids" in batch.keys()
            # compare word ids with those of the tokenizer
            for i, word_ids in enumerate(batch["word_ids"]):
                target_word_ids = tokenized_texts[i].word_ids()
                target_word_ids = [
                    (i if i is not None else -1) for i in target_word_ids
                ]
                assert word_ids == target_word_ids
            # remove from expected keys
            expected_keys.remove("word_ids")

        # check output
        for key in expected_keys:
            assert key in batch.keys()
            # compare with tokenizer output
            for i, x in enumerate(batch[key]):
                assert x == tokenized_texts[i][key]

        # convert to pyarrow table to check whether the
        # output matches the output features
        pa.table(data=batch, schema=p.out_features.arrow_schema)
