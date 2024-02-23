import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.sequence.apply_mask import (
    ApplyMask,
    ApplyMaskConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestApplyMask(BaseTestDataProcessor):
    @pytest.fixture
    def length(self):
        return 5

    @pytest.fixture(params=list(range(5)))
    def mask(self, request, length):
        return [i != request.param for i in range(length)]

    @pytest.fixture
    def in_features(self, length):
        return Features(
            {
                "mask": Sequence(Value("bool"), length=length),
                "seqA": Sequence(Value("int32"), length=length),
                "seqB": Sequence(Value("string"), length=length),
            }
        )

    @pytest.fixture
    def processor(self):
        return ApplyMask(
            ApplyMaskConfig(
                mask="mask", sequences={"seqA": "seqA", "seqB": "seqB"}
            )
        )

    @pytest.fixture
    def in_batch(self, mask, length):
        return {
            "mask": [mask],
            "seqA": [list(range(length))],
            "seqB": [list(map(str, range(length)))],
        }

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "seqA": Sequence(Value("int32")),
                "seqB": Sequence(Value("string")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, length, mask):
        return {
            "seqA": [[x for x, v in zip(range(length), mask) if v]],
            "seqB": [[str(x) for x, v in zip(range(length), mask) if v]],
        }


class TestApplyMaskPreparationErrors(BaseTestDataProcessor):
    @pytest.fixture(params=[[-1, 5], [5, 4]])
    def mask_and_seq_length(self, request):
        return request.param

    @pytest.fixture
    def mask_length(self, mask_and_seq_length):
        return mask_and_seq_length[0]

    @pytest.fixture
    def seq_length(self, mask_and_seq_length):
        return mask_and_seq_length[1]

    @pytest.fixture
    def in_features(self, mask_length, seq_length):
        return Features(
            {
                "mask": Sequence(Value("bool"), length=mask_length),
                "seqA": Sequence(Value("int32"), length=seq_length),
                "seqB": Sequence(Value("string"), length=seq_length),
            }
        )

    @pytest.fixture
    def processor(self):
        return ApplyMask(
            ApplyMaskConfig(
                mask="mask", sequences={"seqA": "seqA", "seqB": "seqB"}
            )
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return TypeError


class TestApplyMaskProcessErrors(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "mask": Sequence(Value("bool")),
                "seqA": Sequence(Value("int32")),
                "seqB": Sequence(Value("string")),
            }
        )

    @pytest.fixture
    def processor(self):
        return ApplyMask(
            ApplyMaskConfig(
                mask="mask", sequences={"seqA": "seqA", "seqB": "seqB"}
            )
        )

    @pytest.fixture
    def in_batch(self):
        return {"mask": [[True] * 10], "seqA": [[1] * 8], "seqB": [["a"] * 6]}

    @pytest.fixture
    def expected_err_on_process(self):
        return ValueError
