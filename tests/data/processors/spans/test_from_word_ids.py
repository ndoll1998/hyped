from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.spans.outputs import SpansOutputs
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
    def in_batch(self, spans):
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
            TokenSpansFromWordIdsConfig(word_ids="word_ids", mask=None)
        )

    @pytest.fixture
    def expected_out_feature(self):
        return Features(
            {
                SpansOutputs.BEGINS: Sequence(Value("int32")),
                SpansOutputs.ENDS: Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, spans):
        # pack features together
        spans_begin, spans_end = zip(*spans)
        # return all features new
        return {
            SpansOutputs.BEGINS: [list(spans_begin)],
            SpansOutputs.ENDS: [list(spans_end)],
        }


class TestTokenSpansFromWordIdsWithMask(TestTokenSpansFromWordIds):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "word_ids": Sequence(Value("int32")),
                "mask": Sequence(Value("bool")),
            }
        )

    @pytest.fixture
    def in_batch(self, spans):
        # create initial word ids sequence of all -1
        length = max(e for _, e in spans)
        word_ids = [-1] * (length + 2)
        # fill with actual word ids from spans
        for i, (b, e) in enumerate(spans):
            word_ids[b + 1 : e + 1] = [i] * (e - b)  # noqa: E203
        # return word ids
        return {
            "word_ids": [word_ids],
            "mask": [[True] + [False] * length + [True]],
        }

    @pytest.fixture
    def processor(self):
        return TokenSpansFromWordIds(
            TokenSpansFromWordIdsConfig(word_ids="word_ids", mask="mask")
        )

    @pytest.fixture
    def expected_out_batch(self, spans):
        # pack features together
        spans_begin, spans_end = zip(*spans)
        # return all features new
        return {
            SpansOutputs.BEGINS: [[b + 1 for b in spans_begin]],
            SpansOutputs.ENDS: [[e + 1 for e in spans_end]],
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
