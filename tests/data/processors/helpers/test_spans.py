from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.helpers.spans import (
    CharToTokenSpans,
    CharToTokenSpansConfig,
)
from datasets import Features, Sequence, Value
import pytest


class TestCharToTokenSpansError(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            # invalid feature type
            {
                "char_spans_begin": Value("string"),
                "char_spans_end": Value("int32"),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("int32")),
            },
            {
                "char_spans_begin": Value("int32"),
                "char_spans_end": Value("string"),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("int32")),
            },
            {
                "char_spans_begin": Sequence(Value("string")),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("int32")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Sequence(Value("string")),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("int32")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("string")),
                "token_offsets_end": Sequence(Value("int32")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("string")),
            },
            # mismatches
            {
                "char_spans_begin": Value("int32"),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("string")),
                "token_offsets_end": Sequence(Value("string")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Value("int32"),
                "token_offsets_begin": Sequence(Value("string")),
                "token_offsets_end": Sequence(Value("string")),
            },
            {
                "char_spans_begin": Sequence(Value("int32"), length=8),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("string")),
                "token_offsets_end": Sequence(Value("string")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Sequence(Value("int32"), length=8),
                "token_offsets_begin": Sequence(Value("string")),
                "token_offsets_end": Sequence(Value("string")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("string"), length=8),
                "token_offsets_end": Sequence(Value("string")),
            },
            {
                "char_spans_begin": Sequence(Value("int32")),
                "char_spans_end": Sequence(Value("int32")),
                "token_offsets_begin": Sequence(Value("string")),
                "token_offsets_end": Sequence(Value("string"), length=8),
            },
        ]
    )
    def in_features(self, request):
        return Features(request.param)

    @pytest.fixture
    def processor(self):
        return CharToTokenSpans(
            CharToTokenSpansConfig(
                char_spans_begin="char_spans_begin",
                char_spans_end="char_spans_end",
                token_offsets_begin="token_offsets_begin",
                token_offsets_end="token_offsets_end",
            )
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return TypeError


class TestCharToTokenSpans(BaseTestDataProcessor):
    @pytest.fixture(params=[True, False])
    def is_char_span_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def is_token_offset_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[-1, 0, 1, 2])
    def num_annotations(self, request):
        return request.param

    @pytest.fixture
    def in_features(self, num_annotations):
        return Features(
            {
                "char_spans_begin": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "char_spans_end": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def batch(
        self,
        num_annotations,
        is_char_span_inclusive,
        is_token_offset_inclusive,
    ):
        a = int(is_char_span_inclusive)
        b = int(is_token_offset_inclusive)

        return {
            "char_spans_begin": [
                [6, 16][:num_annotations],
                [16, 6][:num_annotations],
            ],
            "char_spans_end": [
                [15 - a, 36 - a][:num_annotations],
                [36 - a, 15 - a][:num_annotations],
            ],
            "token_offsets_begin": [
                [0, 6, 10, 16, 24, 31],
                [0, 6, 10, 16, 24, 31],
            ],
            "token_offsets_end": [
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
            ],
        }

    @pytest.fixture
    def processor(self, is_char_span_inclusive, is_token_offset_inclusive):
        return CharToTokenSpans(
            CharToTokenSpansConfig(
                char_spans_begin="char_spans_begin",
                char_spans_end="char_spans_end",
                token_offsets_begin="token_offsets_begin",
                token_offsets_end="token_offsets_end",
                char_spans_inclusive=is_char_span_inclusive,
                token_offsets_inclusive=is_token_offset_inclusive,
            )
        )

    @pytest.fixture
    def expected_out_features(self, num_annotations):
        return Features(
            {
                "token_spans_begin": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "token_spans_end": Sequence(
                    Value("int32"), length=num_annotations
                ),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, num_annotations):
        return {
            "token_spans_begin": [
                [1, 3][:num_annotations],
                [3, 1][:num_annotations],
            ],
            "token_spans_end": [
                [2, 5][:num_annotations],
                [5, 2][:num_annotations],
            ],
        }


class TestSingleCharToTokenSpan(TestCharToTokenSpans):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "char_spans_begin": Value("int32"),
                "char_spans_end": Value("int32"),
                "token_offsets_begin": Sequence(Value("int32")),
                "token_offsets_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def batch(self, is_char_span_inclusive, is_token_offset_inclusive):
        a = int(is_char_span_inclusive)
        b = int(is_token_offset_inclusive)

        return {
            "char_spans_begin": [6, 16],
            "char_spans_end": [15 - a, 36 - a],
            "token_offsets_begin": [
                [0, 6, 10, 16, 24, 31],
                [0, 6, 10, 16, 24, 31],
            ],
            "token_offsets_end": [
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
            ],
        }

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "token_spans_begin": Value("int32"),
                "token_spans_end": Value("int32"),
            }
        )

    @pytest.fixture
    def expected_out_batch(self):
        return {
            "token_spans_begin": [1, 3],
            "token_spans_end": [2, 5],
        }
