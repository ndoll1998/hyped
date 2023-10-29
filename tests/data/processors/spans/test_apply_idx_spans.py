from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.spans.apply_idx_spans import (
    ApplyIndexSpans,
    ApplyIndexSpansConfig,
)
from datasets import Features, Sequence, Value
import pytest


class TestApplyIndexSpans(BaseTestDataProcessor):
    @pytest.fixture(params=[True, False])
    def is_idx_spans_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def is_spans_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[-1, 0, 1, 2])
    def num_annotations(self, request):
        return request.param

    @pytest.fixture
    def in_features(self, num_annotations):
        return Features(
            {
                "idx_spans_begin": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "idx_spans_end": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def batch(
        self,
        num_annotations,
        is_idx_spans_inclusive,
        is_spans_inclusive,
    ):
        a = int(is_idx_spans_inclusive)
        b = int(is_spans_inclusive)

        return {
            "idx_spans_begin": [
                [1, 5][:num_annotations],
                [5, 1][:num_annotations],
            ],
            "idx_spans_end": [
                [4 - a, 6 - a][:num_annotations],
                [6 - a, 4 - a][:num_annotations],
            ],
            "spans_begin": [
                [0, 6, 10, 16, 24, 31],
                [0, 6, 10, 16, 24, 31],
            ],
            "spans_end": [
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
            ],
        }

    @pytest.fixture
    def processor(self, is_idx_spans_inclusive, is_spans_inclusive):
        return ApplyIndexSpans(
            ApplyIndexSpansConfig(
                idx_spans_begin="idx_spans_begin",
                idx_spans_end="idx_spans_end",
                spans_begin="spans_begin",
                spans_end="spans_end",
                is_idx_spans_inclusive=is_idx_spans_inclusive,
                is_spans_inclusive=is_spans_inclusive,
            )
        )

    @pytest.fixture
    def expected_out_features(self, num_annotations):
        return Features(
            {
                "spans_begin": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "spans_end": Sequence(Value("int32"), length=num_annotations),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, num_annotations):
        return {
            "spans_begin": [
                [6, 31][:num_annotations],
                [31, 6][:num_annotations],
            ],
            "spans_end": [
                [23, 36][:num_annotations],
                [36, 23][:num_annotations],
            ],
        }


class TestApplySingleIndexSpan(TestApplyIndexSpans):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "idx_spans_begin": Value("int32"),
                "idx_spans_end": Value("int32"),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def batch(
        self,
        is_idx_spans_inclusive,
        is_spans_inclusive,
    ):
        a = int(is_idx_spans_inclusive)
        b = int(is_spans_inclusive)

        return {
            "idx_spans_begin": [1, 5],
            "idx_spans_end": [4 - a, 6 - a],
            "spans_begin": [
                [0, 6, 10, 16, 24, 31],
                [0, 6, 10, 16, 24, 31],
            ],
            "spans_end": [
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
                [5 - b, 9 - b, 15 - b, 23 - b, 30 - b, 36 - b],
            ],
        }

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "spans_begin": Value("int32"),
                "spans_end": Value("int32"),
            }
        )

    @pytest.fixture
    def expected_out_batch(self):
        return {
            "spans_begin": [6, 31],
            "spans_end": [23, 36],
        }
