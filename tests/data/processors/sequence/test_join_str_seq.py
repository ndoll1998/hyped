import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.sequence.join_str_seq import (
    JoinStringSequence,
    JoinStringSequenceConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestJoinStringSequence(BaseTestDataProcessor):
    @pytest.fixture(params=[" ", ",", " | "])
    def delimiter(self, request):
        return request.param

    @pytest.fixture
    def in_features(self):
        return Features({"seq": Sequence(Value("string"))})

    @pytest.fixture
    def processor(self, delimiter):
        return JoinStringSequence(
            JoinStringSequenceConfig(sequence="seq", delimiter=delimiter)
        )

    @pytest.fixture
    def in_batch(self):
        return {
            "seq": [
                ["apple", "orange", "banana"],
                ["cat", "dog", "fish"],
                ["red", "green", "blue"],
                ["python", "java", "c++"],
                ["sun", "moon", "stars"],
            ]
        }

    @pytest.fixture
    def expected_out_features(self):
        return Features({"joined_string": Value("string")})

    @pytest.fixture
    def expected_out_batch(self, delimiter, in_batch):
        return {
            "joined_string": [delimiter.join(seq) for seq in in_batch["seq"]]
        }
