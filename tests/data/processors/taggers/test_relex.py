import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.taggers.relex import (
    RelExTagger,
    RelExTaggerConfig,
    RelExTaggerOutputs,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestRelExTagger(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "source_span": (1, 3),
                "target_span": (4, 7),
                "expected_out": [0, -1, 1, 2, -2, 3, -3, 4, 5, 6, -4, 7, 8],
                "max_length": None,
            },
            # spans reach sequence limit
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "source_span": (0, 3),
                "target_span": (4, 7),
                "expected_out": [-1, 0, 1, 2, -2, 3, -3, 4, 5, 6, -4, 7, 8],
                "max_length": None,
            },
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "source_span": (1, 3),
                "target_span": (4, 9),
                "expected_out": [0, -1, 1, 2, -2, 3, -3, 4, 5, 6, 7, 8, -4],
                "max_length": None,
            },
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "source_span": (0, 3),
                "target_span": (4, 9),
                "expected_out": [-1, 0, 1, 2, -2, 3, -3, 4, 5, 6, 7, 8, -4],
                "max_length": None,
            },
            # truncate to maximum sequence length
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "source_span": (5, 6),
                "target_span": (6, 8),
                "expected_out": [4, -1, 5, -2, -3, 6, 7, -4, 8, 9],
                "max_length": 10,
            },
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "source_span": (9, 10),
                "target_span": (11, 12),
                "expected_out": [6, 7, 8, -1, 9, -2, 10, -3, 11, -4, 12],
                "max_length": 11,
            },
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "source_span": (2, 3),
                "target_span": (4, 5),
                "expected_out": [-1, 2, -2, 3, -3, 4, -4, 5],
                "max_length": 8,
            },
            # to long to fit into sequence length
            {
                "input_sequence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "source_span": (1, 3),
                "target_span": (6, 11),
                "expected_out": None,
                "max_length": 8,
            },
        ]
    )
    def example(self, request):
        return request.param

    @pytest.fixture
    def length(self, example):
        return len(example["input_sequence"])

    @pytest.fixture
    def max_length(self, example):
        return example["max_length"]

    @pytest.fixture
    def out_length(self, example, max_length, length):
        if max_length is None:
            return length + 4

        return min(max_length, length + 4)

    @pytest.fixture(params=[False, True])
    def is_source_span_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[False, True])
    def is_target_span_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[int, str])
    def value_type(self, request):
        return request.param

    @pytest.fixture
    def sequence_feature(self, value_type):
        return Value({int: "int32", str: "string"}.get(value_type))

    @pytest.fixture
    def in_features(self, sequence_feature, length):
        return Features(
            {
                "input_sequence": Sequence(sequence_feature, length=length),
                "source_span_begin": Value("int32"),
                "source_span_end": Value("int32"),
                "target_span_begin": Value("int32"),
                "target_span_end": Value("int32"),
            }
        )

    @pytest.fixture
    def processor(
        self,
        value_type,
        is_source_span_inclusive,
        is_target_span_inclusive,
        max_length,
    ):
        # build markers
        markers = list(map(value_type, [-1, -2, -3, -4]))
        # build relation extraction tagger
        return RelExTagger(
            RelExTaggerConfig(
                source_begin_marker=markers[0],
                source_end_marker=markers[1],
                target_begin_marker=markers[2],
                target_end_marker=markers[3],
                input_sequence="input_sequence",
                source_span_begin="source_span_begin",
                source_span_end="source_span_end",
                target_span_begin="target_span_begin",
                target_span_end="target_span_end",
                source_span_inclusive=is_source_span_inclusive,
                target_span_inclusive=is_target_span_inclusive,
                max_sequence_length=max_length,
            )
        )

    @pytest.fixture
    def in_batch(
        self,
        value_type,
        example,
        is_source_span_inclusive,
        is_target_span_inclusive,
    ):
        return {
            "input_sequence": [
                list(map(value_type, example["input_sequence"]))
            ],
            "source_span_begin": [example["source_span"][0]],
            "source_span_end": [
                example["source_span"][1] - int(is_source_span_inclusive)
            ],
            "target_span_begin": [example["target_span"][0]],
            "target_span_end": [
                example["target_span"][1] - int(is_target_span_inclusive)
            ],
        }

    @pytest.fixture
    def expected_out_features(self, sequence_feature, out_length):
        return {
            RelExTaggerOutputs.MARKED_SEQUENCE: Sequence(
                sequence_feature, length=out_length
            )
        }

    @pytest.fixture
    def expected_out_batch(self, example, value_type):
        if example["expected_out"] is not None:
            return {
                RelExTaggerOutputs.MARKED_SEQUENCE: [
                    list(map(value_type, example["expected_out"]))
                ]
            }

        else:
            return {RelExTaggerOutputs.MARKED_SEQUENCE: []}
