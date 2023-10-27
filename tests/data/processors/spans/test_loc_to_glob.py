from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.spans.loc_to_glob import (
    LocalToGlobalOffsets,
    LocalToGlobalOffsetsConfig,
)
from datasets import Features, Sequence, Value
import pytest


class TestLocalToGlobalOffsetsKeyErrors(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "local_offsets_begin": Sequence(Value("int32"), length=8),
                "local_offsets_end": Sequence(Value("int32"), length=8),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Sequence(Value("int32"), length=16),
            }
        )

    @pytest.fixture(
        params=[
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="INVALID_KEY",
                local_offsets_end="local_offsets_end",
                global_offsets_begin="global_offsets_begin",
                local_to_global_mapping="local_to_global_mapping",
            ),
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="local_offsets_begin",
                local_offsets_end="INVALID_KEY",
                global_offsets_begin="global_offsets_begin",
                local_to_global_mapping="local_to_global_mapping",
            ),
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="local_offsets_begin",
                local_offsets_end="local_offsets_end",
                global_offsets_begin="INVALID_KEY",
                local_to_global_mapping="local_to_global_mapping",
            ),
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="local_offsets_begin",
                local_offsets_end="local_offsets_end",
                global_offsets_begin="global_offsets_begin",
                local_to_global_mapping="INVALID_KEY",
            ),
        ]
    )
    def processor(self, request):
        return LocalToGlobalOffsets(request.param)

    @pytest.fixture
    def expected_err_on_prepare(self):
        return KeyError


class TestLocalToGlobalOffsetsTypeErrors(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            # invalid types
            {
                "local_offsets_begin": Value("int32"),
                "local_offsets_end": Sequence(Value("int32")),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Sequence(Value("int32")),
            },
            {
                "local_offsets_begin": Sequence(Value("int32")),
                "local_offsets_end": Value("int32"),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Sequence(Value("int32")),
            },
            {
                "local_offsets_begin": Sequence(Value("int32")),
                "local_offsets_end": Sequence(Value("int32")),
                "global_offsets_begin": Value("int32"),
                "local_to_global_mapping": Sequence(Value("int32")),
            },
            {
                "local_offsets_begin": Sequence(Value("int32")),
                "local_offsets_end": Sequence(Value("int32")),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Value("int32"),
            },
            # mismatching types
            {
                "local_offsets_begin": Sequence(Value("int32"), length=8),
                "local_offsets_end": Sequence(Value("int32"), length=16),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Sequence(Value("int32")),
            },
            {
                "local_offsets_begin": Sequence(Value("int32"), length=8),
                "local_offsets_end": Sequence(Value("int32"), length=8),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Sequence(Value("int32"), length=16),
            },
        ]
    )
    def in_features(self, request):
        return Features(request.param)

    @pytest.fixture
    def processor(self):
        return LocalToGlobalOffsets(
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="local_offsets_begin",
                local_offsets_end="local_offsets_end",
                global_offsets_begin="global_offsets_begin",
                local_to_global_mapping="local_to_global_mapping",
            )
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return TypeError


class TestLocalToGlobalOffsets(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "local_offsets_begin": Sequence(Value("int32")),
                "local_offsets_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def processor(self):
        return LocalToGlobalOffsets(
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="local_offsets_begin",
                local_offsets_end="local_offsets_end",
            )
        )

    @pytest.fixture
    def batch(self):
        return {
            "local_offsets_begin": [[0, 4, 6, 0, 3, 5, 0, 3]],
            "local_offsets_end": [[4, 6, 9, 3, 5, 9, 3, 7]],
        }

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "offsets_begin": Sequence(Value("int32")),
                "offsets_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self):
        return {
            "offsets_begin": [[0, 4, 6, 10, 13, 15, 20, 23]],
            "offsets_end": [[4, 6, 9, 13, 15, 19, 23, 27]],
        }


class TestLocalToGlobalOffsetsWithGlobalOffsets(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "local_offsets_begin": Sequence(Value("int32")),
                "local_offsets_end": Sequence(Value("int32")),
                "global_offsets_begin": Sequence(Value("int32")),
                "local_to_global_mapping": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def processor(self):
        return LocalToGlobalOffsets(
            LocalToGlobalOffsetsConfig(
                local_offsets_begin="local_offsets_begin",
                local_offsets_end="local_offsets_end",
                global_offsets_begin="global_offsets_begin",
                local_to_global_mapping="local_to_global_mapping",
            )
        )

    @pytest.fixture
    def batch(self):
        return {
            "local_offsets_begin": [[0, 4, 6, 0, 3, 5, 0, 3]],
            "local_offsets_end": [[4, 6, 9, 3, 5, 9, 3, 7]],
            "local_to_global_mapping": [[0, 0, 0, 1, 1, 1, 2, 2]],
            "global_offsets_begin": [[16, 32, 64]],
        }

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "offsets_begin": Sequence(Value("int32")),
                "offsets_end": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self):
        return {
            "offsets_begin": [[16, 20, 22, 32, 35, 37, 64, 67]],
            "offsets_end": [[20, 22, 25, 35, 37, 41, 67, 71]],
        }
