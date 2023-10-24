from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.features.flatten import (
    FlattenFeatures,
    FlattenFeaturesConfig,
)
from datasets import Features, Sequence, Value
import pytest


class BaseTestFlattenFeatures(BaseTestDataProcessor):
    @pytest.fixture()
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

    @pytest.fixture()
    def batch(self):
        return {
            "X": list(range(0, 12)),
            "Y": [[i + 1, i + 2, i + 3] for i in range(0, 12)],
            "A": [{"x": 2 * i, "y": [3 * i, 4 * i]} for i in range(0, 12)],
        }


class TestFlattenAllFeatures(BaseTestFlattenFeatures):
    @pytest.fixture
    def processor(self):
        return FlattenFeatures(FlattenFeaturesConfig(delimiter="."))

    @pytest.fixture
    def expected_out_features(self):
        return Features(
            {
                "X": Value("int32"),
                # flatten sequence
                "Y.0": Value("int32"),
                "Y.1": Value("int32"),
                "Y.2": Value("int32"),
                # flatten mapping
                "A.x": Value("int32"),
                "A.y.0": Value("int32"),
                "A.y.1": Value("int32"),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, batch):
        return {
            "X": batch["X"],
            "Y.0": [x for x, _, _ in batch["Y"]],
            "Y.1": [x for _, x, _ in batch["Y"]],
            "Y.2": [x for _, _, x in batch["Y"]],
            "A.x": [item["x"] for item in batch["A"]],
            "A.y.0": [item["y"][0] for item in batch["A"]],
            "A.y.1": [item["y"][1] for item in batch["A"]],
        }


class TestFlattenSelectedFeatures(BaseTestFlattenFeatures):
    @pytest.fixture(
        params=[
            [],
            ["X"],
            ["Y"],
            ["A"],
            ["X", "Y"],
            ["X", "A"],
            ["A", "Y"],
            ["X", "Y", "A"],
        ]
    )
    def processor(self, request):
        return FlattenFeatures(
            FlattenFeaturesConfig(delimiter=".", to_flatten=request.param)
        )

    @pytest.fixture
    def expected_out_features(self, processor):
        features = Features()

        if "X" in processor.config.to_flatten:
            features["X"] = Value("int32")

        if "Y" in processor.config.to_flatten:
            features["Y.0"] = Value("int32")
            features["Y.1"] = Value("int32")
            features["Y.2"] = Value("int32")

        if "A" in processor.config.to_flatten:
            features["A.x"] = Value("int32")
            features["A.y.0"] = Value("int32")
            features["A.y.1"] = Value("int32")

        return features

    @pytest.fixture
    def expected_out_batch(self, batch, processor):
        out_batch = {}

        if "X" in processor.config.to_flatten:
            out_batch["X"] = batch["X"]

        if "Y" in processor.config.to_flatten:
            out_batch["Y.0"] = [x for x, _, _ in batch["Y"]]
            out_batch["Y.1"] = [x for _, x, _ in batch["Y"]]
            out_batch["Y.2"] = [x for _, _, x in batch["Y"]]

        if "A" in processor.config.to_flatten:
            out_batch["A.x"] = [item["x"] for item in batch["A"]]
            out_batch["A.y.0"] = [item["y"][0] for item in batch["A"]]
            out_batch["A.y.1"] = [item["y"][1] for item in batch["A"]]

        return out_batch


class TestFeatureKeyNotFound(BaseTestFlattenFeatures):
    @pytest.fixture
    def processor(self):
        return FlattenFeatures(
            FlattenFeaturesConfig(to_flatten=["INVALID_KEY"])
        )

    @pytest.fixture
    def expected_err_on_prepare(self):
        return KeyError
