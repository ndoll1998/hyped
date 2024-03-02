import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.data.processors.sequence.filter import (
    FilterSequence,
    FilterSequenceConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestFilterSequence(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            [int(i < 3) for i in range(10)],
            [int(i < 5) for i in range(10)],
            [int(i < 7) for i in range(10)],
            [int(i % 2) for i in range(10)],
            [int(i % 3) for i in range(10)],
            [int(i % 4) for i in range(10)],
            [int(i % 5) for i in range(10)],
        ]
    )
    def example(self, request):
        return request.param

    @pytest.fixture
    def in_features(self):
        return Features({"sequence": Sequence(Value("int32"))})

    @pytest.fixture
    def in_batch(self, example):
        return {"sequence": [example]}

    @pytest.fixture
    def processor(self):
        return FilterSequence(
            FilterSequenceConfig(sequence="sequence", valids=[0])
        )

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "filtered_sequence": Sequence(ClassLabel(names=[0])),
                "filter_mask": Sequence(Value("bool")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, example):
        return {
            "filtered_sequence": [[0 for i in example if i == 0]],
            "filter_mask": [[i == 0 for i in example]],
        }
