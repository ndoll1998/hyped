import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.sequence.extend import (
    ExtendSequence,
    ExtendSequenceConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestExtendSequence(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            {
                "feature": Value("int32"),
                "sequence": list(range(10)),
                "length": -1,
                "prepend": [-1],
                "append": [-2],
            },
            {
                "feature": Value("int32"),
                "sequence": list(range(10)),
                "length": 10,
                "prepend": [-1],
                "append": [-2],
            },
            {
                "feature": Value("int32"),
                "sequence": list(range(32)),
                "length": -1,
                "prepend": list(range(0, 64)),
                "append": list(range(64, -1, -1)),
            },
            {
                "feature": Value("int32"),
                "sequence": list(range(32)),
                "length": 32,
                "prepend": list(range(0, 64)),
                "append": list(range(64, -1, -1)),
            },
            {
                "feature": Sequence(Value("int32")),
                "sequence": [[i] for i in range(32)],
                "length": 32,
                "prepend": [[-1], [-2]],
                "append": [[-3], [-4]],
            },
            {
                "feature": Sequence(Value("int32"), length=3),
                "sequence": [[i, i, i] for i in range(32)],
                "length": 32,
                "prepend": [[-1, -1, -1]],
                "append": [[-2, -2, -2]],
            },
            {
                "feature": Sequence(Value("int32"), length=3),
                "sequence": [[i, i, i] for i in range(32)],
                "length": 32,
                "prepend": [[-i, -i, -i] for i in range(64)],
                "append": [[-i, -i, -i] for i in range(64, 128)],
            },
        ]
    )
    def setup(self, request):
        return request.param

    @pytest.fixture
    def feature(self, setup):
        return setup["feature"]

    @pytest.fixture
    def sequence(self, setup):
        return setup["sequence"]

    @pytest.fixture
    def length(self, setup):
        return setup["length"]

    @pytest.fixture
    def prepend(self, setup):
        return setup["prepend"]

    @pytest.fixture
    def append(self, setup):
        return setup["append"]

    @pytest.fixture
    def in_features(self, feature, length):
        return Features({"sequence": Sequence(feature, length=length)})

    @pytest.fixture
    def processor(self, append, prepend):
        return ExtendSequence(
            ExtendSequenceConfig(
                sequence="sequence",
                output="sequence",
                append=append,
                prepend=prepend,
            )
        )

    @pytest.fixture
    def in_batch(self, sequence):
        return {"sequence": [sequence]}

    @pytest.fixture
    def expected_out_features(self, feature, length, append, prepend):
        if length != -1:
            length += len(append) + len(prepend)

        return Features({"sequence": Sequence(feature, length=length)})

    @pytest.fixture
    def expected_out_batch(self, append, prepend, sequence):
        return {"sequence": [prepend + sequence + append]}
