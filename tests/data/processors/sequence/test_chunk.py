import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.sequence.chunk import (
    ChunkSequence,
    ChunkSequenceConfig,
)
from tests.data.processors.base import BaseTestDataProcessor


class TestChunkSequence(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            {
                # perfect chunk no overlap
                "length": 8,
                "chunk_size": 4,
                "chunk_stride": 4,
                "drop_last": False,
                "expected_out_length": 4,
                "chunk_spans": [(0, 4), (4, 8)],
            },
            {
                # perfect chunk no overlap
                "length": 8,
                "chunk_size": 4,
                "chunk_stride": 4,
                "drop_last": True,
                "expected_out_length": 4,
                "chunk_spans": [(0, 4), (4, 8)],
            },
            {
                # perfect chunk with overlap
                "length": 8,
                "chunk_size": 4,
                "chunk_stride": 2,
                "drop_last": False,
                "expected_out_length": 4,
                "chunk_spans": [(0, 4), (2, 6), (4, 8)],
            },
            {
                # perfect chunk with overlap
                "length": 8,
                "chunk_size": 4,
                "chunk_stride": 2,
                "drop_last": True,
                "expected_out_length": 4,
                "chunk_spans": [(0, 4), (2, 6), (4, 8)],
            },
            {
                # no perfect chunk keep remainder
                "length": 8,
                "chunk_size": 4,
                "chunk_stride": 3,
                "drop_last": False,
                "expected_out_length": -1,
                "chunk_spans": [(0, 4), (3, 7), (6, 8)],
            },
            {
                # no perfect chunk drop remainder
                "length": 8,
                "chunk_size": 4,
                "chunk_stride": 3,
                "drop_last": True,
                "expected_out_length": 4,
                "chunk_spans": [(0, 4), (3, 7)],
            },
        ]
    )
    def setup(self, request):
        return request.param

    @pytest.fixture
    def length(self, setup):
        return setup["length"]

    @pytest.fixture
    def chunk_size(self, setup):
        return setup["chunk_size"]

    @pytest.fixture
    def chunk_stride(self, setup):
        return setup["chunk_stride"]

    @pytest.fixture
    def drop_last(self, setup):
        return setup["drop_last"]

    @pytest.fixture
    def expected_out_length(self, setup):
        return setup["expected_out_length"]

    @pytest.fixture
    def chunk_spans(self, setup):
        return setup["chunk_spans"]

    @pytest.fixture
    def in_features(self, length):
        return Features(
            {
                "seqA": Sequence(Value("int32"), length),
                "seqB": Sequence(Value("int32"), length),
            }
        )

    @pytest.fixture
    def in_batch(self, length):
        return {
            "seqA": [list(range(length))],
            "seqB": [list(range(-5, -5 + length))],
        }

    @pytest.fixture
    def processor(self, chunk_size, chunk_stride, drop_last):
        return ChunkSequence(
            ChunkSequenceConfig(
                sequence=["seqA", "seqB"],
                chunk_size=chunk_size,
                chunk_stride=chunk_stride,
                drop_last=drop_last,
            )
        )

    @pytest.fixture
    def expected_out_features(self, expected_out_length):
        return Features(
            {
                "seqA": Sequence(Value("int32"), length=expected_out_length),
                "seqB": Sequence(Value("int32"), length=expected_out_length),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, in_batch, chunk_spans):
        return {
            "seqA": [in_batch["seqA"][0][b:e] for b, e in chunk_spans],
            "seqB": [in_batch["seqB"][0][b:e] for b, e in chunk_spans],
        }
