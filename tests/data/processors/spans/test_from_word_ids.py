from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.spans.from_word_ids import (
    TokenSpansFromWordIds,
    TokenSpansFromWordIdsConfig,
)
from datasets import Features, Sequence, Value
import pytest


class TestTokenSpansFromWordIds(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            [(0, 5)],
            [(0, 5), (5, 10)],
            [(0, 5), (5, 10), (10, 13)],
            [(0, 5), (5, 10), (10, 13), (13, 18)],
        ]
    )
    def spans(self, request):
        return request.param

    @pytest.fixture
    def in_features(self):
        return Features({"word_ids": Sequence(Value("int32"))})

    @pytest.fixture
    def batch(self, spans):
        # create initial word ids sequence of all -1
        length = max(e for _, e in spans)
        word_ids = [-1] * length
        # fill with actual word ids from spans
        for i, (b, e) in enumerate(spans):
            word_ids[b:e] = [i] * (e - b)
        # return word ids
        return {"word_ids": [word_ids]}

    @pytest.fixture
    def processor(self):
        return TokenSpansFromWordIds(
            TokenSpansFromWordIdsConfig(word_ids="word_ids")
        )

    @pytest.fixture
    def expected_out_feature(self):
        return Features(
            {
                "token_spans_begin": Sequence(Value("int32")),
                "token_spans_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, spans):
        # pack features together
        spans_begin, spans_end = zip(*spans)
        # return all features new
        return {
            "token_spans_begin": [list(spans_begin)],
            "token_spans_end": [list(spans_end)],
        }


class TestErrorOnInvalidWordIds(TestTokenSpansFromWordIds):
    @pytest.fixture(
        params=[
            # holes in word id sequence
            [(0, 5), (6, 10)],
            [(0, 5), (5, 10), (11, 13)],
            [(0, 5), (5, 9), (10, 13), (13, 18)],
        ]
    )
    def spans(self, request):
        return request.param

    @pytest.fixture
    def expected_err_on_process(self):
        return ValueError
