import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.spans.idx_spans import (
    CoveredIndexSpans,
    CoveredIndexSpansConfig,
)
from hyped.data.processors.spans.outputs import SpansOutputs
from tests.data.processors.base import BaseTestDataProcessor


class TestCoveredIndexSpansErrors(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            # invalid feature type
            {
                "queries_begin": Value("string"),
                "queries_end": Value("int32"),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            },
            {
                "queries_begin": Value("int32"),
                "queries_end": Value("string"),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            },
            {
                "queries_begin": Sequence(Value("string")),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Sequence(Value("string")),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("string")),
                "spans_end": Sequence(Value("int32")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("string")),
            },
            # mismatches
            {
                "queries_begin": Value("int32"),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("string")),
                "spans_end": Sequence(Value("string")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Value("int32"),
                "spans_begin": Sequence(Value("string")),
                "spans_end": Sequence(Value("string")),
            },
            {
                "queries_begin": Sequence(Value("int32"), length=8),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("string")),
                "spans_end": Sequence(Value("string")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Sequence(Value("int32"), length=8),
                "spans_begin": Sequence(Value("string")),
                "spans_end": Sequence(Value("string")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("string"), length=8),
                "spans_end": Sequence(Value("string")),
            },
            {
                "queries_begin": Sequence(Value("int32")),
                "queries_end": Sequence(Value("int32")),
                "spans_begin": Sequence(Value("string")),
                "spans_end": Sequence(Value("string"), length=8),
            },
        ]
    )
    def in_features(self, request):
        return Features(request.param)

    @pytest.fixture
    def processor(self):
        return CoveredIndexSpans(
            CoveredIndexSpansConfig(
                queries_begin="queries_begin",
                queries_end="queries_end",
                spans_begin="spans_begin",
                spans_end="spans_end",
            )
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return TypeError


class TestCoveredIndexSpans(BaseTestDataProcessor):
    @pytest.fixture(params=[True, False])
    def is_queries_inclusive(self, request):
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
                "queries_begin": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "queries_end": Sequence(
                    Value("int32"), length=num_annotations
                ),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def in_batch(
        self,
        num_annotations,
        is_queries_inclusive,
        is_spans_inclusive,
    ):
        a = int(is_queries_inclusive)
        b = int(is_spans_inclusive)

        return {
            "queries_begin": [
                [6, 16][:num_annotations],
                [16, 6][:num_annotations],
            ],
            "queries_end": [
                [15 - a, 36 - a][:num_annotations],
                [36 - a, 15 - a][:num_annotations],
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
    def processor(self, is_queries_inclusive, is_spans_inclusive):
        return CoveredIndexSpans(
            CoveredIndexSpansConfig(
                queries_begin="queries_begin",
                queries_end="queries_end",
                spans_begin="spans_begin",
                spans_end="spans_end",
                is_queries_inclusive=is_queries_inclusive,
                is_spans_inclusive=is_spans_inclusive,
            )
        )

    @pytest.fixture
    def expected_out_features(self, num_annotations):
        return Features(
            {
                SpansOutputs.BEGINS: Sequence(
                    Value("int32"), length=num_annotations
                ),
                SpansOutputs.ENDS: Sequence(
                    Value("int32"), length=num_annotations
                ),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, num_annotations):
        return {
            SpansOutputs.BEGINS: [
                [1, 3][:num_annotations],
                [3, 1][:num_annotations],
            ],
            SpansOutputs.ENDS: [
                [3, 6][:num_annotations],
                [6, 3][:num_annotations],
            ],
        }


class TestSingleCoveredIndexSpan(TestCoveredIndexSpans):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "queries_begin": Value("int32"),
                "queries_end": Value("int32"),
                "spans_begin": Sequence(Value("int32")),
                "spans_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def in_batch(self, is_queries_inclusive, is_spans_inclusive):
        a = int(is_queries_inclusive)
        b = int(is_spans_inclusive)

        return {
            "queries_begin": [6, 16],
            "queries_end": [15 - a, 36 - a],
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
                SpansOutputs.BEGINS: Value("int32"),
                SpansOutputs.ENDS: Value("int32"),
            }
        )

    @pytest.fixture
    def expected_out_batch(self):
        return {
            SpansOutputs.BEGINS: [1, 3],
            SpansOutputs.ENDS: [3, 6],
        }
