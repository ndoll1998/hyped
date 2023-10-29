from hyped.data.processors.features.format import (
    FormatFeaturesConfig,
    FormatFeatures,
)
from tests.data.processors.base import BaseTestDataProcessor
from datasets import Features, Sequence, Value
import pytest


class BaseTestFormatFlatFeatures(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {"X": Value("int32"), "Y": Value("int32"), "A": Value("string")}
        )

    @pytest.fixture
    def batch(self):
        return {
            "X": list(range(0, 12)),
            "Y": list(range(12, 24)),
            "A": list(map(str, range(0, 12))),
        }


class TestRenameFlatFeatures(BaseTestFormatFlatFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(mapping={"new_X": "X", "new_Y": "Y"})
        )

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "new_X": Value("int32"),
                "new_Y": Value("int32"),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "new_X": batch["X"],
            "new_Y": batch["Y"],
        }


class TestPackFlatFeaturesInSequence(BaseTestFormatFlatFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(FormatFeaturesConfig(mapping={"XY": ["X", "Y"]}))

    @pytest.fixture
    def expected_out_features(self):
        return Features({"XY": Sequence(Value("int32"), length=2)})

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {"XY": list(map(list, zip(batch["X"], batch["Y"])))}


class TestErrorOnPackDifferentTypesInSequence(BaseTestFormatFlatFeatures):
    @pytest.fixture
    def processor(self):
        # pack integer and string feature in a sequence is invalid
        return FormatFeatures(FormatFeaturesConfig(mapping={"XA": ["X", "A"]}))

    @pytest.fixture
    def expected_err_on_prepare(self):
        return TypeError


class TestPackFlatFeaturesInDict(BaseTestFormatFlatFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={"XYA": {"X": "X", "YA": {"Y": "Y", "A": "A"}}}
            )
        )

    @pytest.fixture
    def expected_out_features(self, in_features):
        return Features(
            {
                "XYA": {
                    "X": Value("int32"),
                    "YA": {"Y": Value("int32"), "A": Value("string")},
                }
            }
        )

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "XYA": [
                {"X": x, "YA": {"Y": y, "A": a}}
                for x, y, a in zip(batch["X"], batch["Y"], batch["A"])
            ]
        }


class TestPackFlatFeaturesInListOfDicts(BaseTestFormatFlatFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "XYA": [{"XorY": "X", "A": "A"}, {"XorY": "Y", "A": "A"}]
                }
            )
        )

    @pytest.fixture
    def expected_out_features(self):
        f = Features(
            {
                "XYA": Sequence(
                    {"XorY": Value("int32"), "A": Value("string")}, length=2
                )
            }
        )
        return f

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "XYA": [
                [
                    {"XorY": x, "A": a},
                    {"XorY": y, "A": a},
                ]
                for x, y, a in zip(batch["X"], batch["Y"], batch["A"])
            ]
        }


class BaseTestFormatNestedFeatures(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "X": Value("int32"),
                "Y": Sequence(Value("int32"), length=3),
                "A": {
                    "x": Value("int32"),
                    "y": Sequence(Value("int32"), length=2),
                },
            }
        )

    @pytest.fixture
    def batch(self):
        return {
            "X": list(range(0, 12)),
            "Y": [[i + 1, i + 2, i + 3] for i in range(0, 12)],
            "A": [{"x": 2 * i, "y": [3 * i, 4 * i]} for i in range(0, 12)],
        }


class TestErrorOnInvalidKeyType(BaseTestFormatNestedFeatures):
    @pytest.fixture(
        params=[
            [{"Y": ("Y", 1.2)}],  # key must be string or int
            [{"Y": ("Y", "x")}],  # index sequence with string
            [{"Y": ("A", 0)}],  # index mapping with int
        ]
    )
    def processor(self, request):
        return FormatFeatures(FormatFeaturesConfig(mapping=request.param))

    @pytest.fixture
    def expected_err_on_prepare(self):
        return TypeError


class TestAccessNestedSequence(BaseTestFormatNestedFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "Y.0": ("Y", 0),
                }
            )
        )

    @pytest.fixture
    def expected_out_features(self):
        return Features({"new_X": Value("int32"), "Y.0": Value("int32")})

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "new_X": batch["X"],
            "Y.0": [x for x, _, _ in batch["Y"]],
        }


class TestErrorOnIndexOutOfRange(BaseTestFormatNestedFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "Y.0": ("Y", 0),
                    "Y.1": ("Y", 1),
                    "Y.2": ("Y", 2),
                    "Y.3": ("Y", 3),
                }
            )
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return IndexError


class TestAccessNestedMapping(BaseTestFormatNestedFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={"new_X": "X", "A.x": ("A", "x"), "A.y": ("A", "y")}
            )
        )

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "new_X": Value("int32"),
                "A.x": Value("int32"),
                "A.y": Sequence(Value("int32"), length=2),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "new_X": batch["X"],
            "A.x": [item["x"] for item in batch["A"]],
            "A.y": [item["y"] for item in batch["A"]],
        }


class TestErrorKeyNotFound(BaseTestFormatNestedFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "A.x": ("A", "x"),
                    "A.y": ("A", "y"),
                    "A.z": ("A", "z"),
                }
            )
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return KeyError


class TestAccessNestedFeatures(BaseTestFormatNestedFeatures):
    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "A.x": ("A", "x"),
                    "A.y.0": ("A", "y", 0),
                    "A.y.1": ("A", "y", 1),
                }
            )
        )

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "new_X": Value("int32"),
                "A.x": Value("int32"),
                "A.y.0": Value("int32"),
                "A.y.1": Value("int32"),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "new_X": batch["X"],
            "A.x": [item["x"] for item in batch["A"]],
            "A.y.0": [item["y"][0] for item in batch["A"]],
            "A.y.1": [item["y"][1] for item in batch["A"]],
        }


class TestFormatSliceFeatures(BaseTestDataProcessor):
    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "X": Value("int32"),
                "A": Sequence(Sequence(Value("int32"), length=3)),
            }
        )

    @pytest.fixture
    def batch(self):
        return {
            "X": list(range(0, 12)),
            "A": [
                [[i + 1, i + 2, i + 3] for i in range(0, 12)]
                for _ in range(12)
            ],
        }

    @pytest.fixture
    def processor(self):
        return FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "A.0": ("A", slice(-1), 0),
                    "A.1": ("A", slice(-1), 1),
                    "A.2": ("A", slice(-1), 2),
                }
            )
        )

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "new_X": Value("int32"),
                "A.0": Sequence(Value("int32")),
                "A.1": Sequence(Value("int32")),
                "A.2": Sequence(Value("int32")),
            }
        )

    @pytest.fixture
    def expected_out_batch(self):
        return {
            "new_X": list(range(0, 12)),
            "A.0": [[i + 1 for i in range(0, 12)] for _ in range(12)],
            "A.1": [[i + 2 for i in range(0, 12)] for _ in range(12)],
            "A.2": [[i + 3 for i in range(0, 12)] for _ in range(12)],
        }
