from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.features.filter import (
    FilterFeatures,
    FilterFeaturesConfig,
)
from datasets import Features, Sequence, Value
import pytest


class BaseTestFilterFeatures(BaseTestDataProcessor):
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


class TestKeepFeatures(BaseTestFilterFeatures):
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
        return FilterFeatures(FilterFeaturesConfig(keep=request.param))

    @pytest.fixture
    def expected_out_features(self, processor, in_features):
        features = Features()

        if "X" in processor.config.keep:
            features["X"] = in_features["X"]

        if "Y" in processor.config.keep:
            features["Y"] = in_features["Y"]

        if "A" in processor.config.keep:
            features["A"] = in_features["A"]

        return features

    @pytest.fixture
    def expected_out_batch(self, batch, processor):
        out_batch = {}

        if "X" in processor.config.keep:
            out_batch["X"] = batch["X"]

        if "Y" in processor.config.keep:
            out_batch["Y"] = batch["Y"]

        if "A" in processor.config.keep:
            out_batch["A"] = batch["A"]

        return out_batch


class TestRemoveFeatures(BaseTestFilterFeatures):
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
        return FilterFeatures(FilterFeaturesConfig(remove=request.param))

    @pytest.fixture
    def expected_out_features(self, processor, in_features):
        features = Features()

        if "X" not in processor.config.remove:
            features["X"] = in_features["X"]

        if "Y" not in processor.config.remove:
            features["Y"] = in_features["Y"]

        if "A" not in processor.config.remove:
            features["A"] = in_features["A"]

        return features

    @pytest.fixture
    def expected_out_batch(self, batch, processor):
        out_batch = {}

        if "X" not in processor.config.remove:
            out_batch["X"] = batch["X"]

        if "Y" not in processor.config.remove:
            out_batch["Y"] = batch["Y"]

        if "A" not in processor.config.remove:
            out_batch["A"] = batch["A"]

        return out_batch


class TestErrorOnInvalidKey(BaseTestFilterFeatures):
    @pytest.fixture(
        params=[
            FilterFeaturesConfig(keep=["INVALID_KEY"]),
            FilterFeaturesConfig(remove=["INVALID_KEY"]),
        ]
    )
    def processor(self, request):
        return FilterFeatures(request.param)

    @pytest.fixture
    def expected_err_on_prepare(self):
        return KeyError


class TestErrorOnInvalidConfig(BaseTestFilterFeatures):
    @pytest.fixture(
        params=[
            FilterFeaturesConfig(),
            FilterFeaturesConfig(keep=["X"], remove=["Y"]),
        ]
    )
    def processor(self, request):
        return FilterFeatures(request.param)

    @pytest.fixture
    def expected_err_on_prepare(self):
        return ValueError
