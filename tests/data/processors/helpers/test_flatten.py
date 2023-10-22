from hyped.data.processors.helpers.flatten import (
    FlattenFeatures,
    FlattenFeaturesConfig,
)
from datasets import Features, Sequence, Value
import pytest


class TestFlattenFeatures:
    @pytest.fixture()
    def batch(self):
        return {
            "X": list(range(0, 12)),
            "Y": [[i + 1, i + 2, i + 3] for i in range(0, 12)],
            "A": [{"x": 2 * i, "y": [3 * i, 4 * i]} for i in range(0, 12)],
        }

    @pytest.fixture()
    def features(self):
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

    def test_flatten_all_features(self, batch, features):
        # create processor
        p = FlattenFeatures(FlattenFeaturesConfig(delimiter="."))
        p.prepare(features)

        # check features
        assert p.new_features == Features(
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

        # apply processor
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        assert batch["X"] == list(range(0, 12))
        assert batch["Y.0"] == list(range(1, 13))
        assert batch["Y.1"] == list(range(2, 14))
        assert batch["Y.2"] == list(range(3, 15))
        assert batch["A.x"] == list(range(0, 24, 2))
        assert batch["A.y.0"] == list(range(0, 36, 3))
        assert batch["A.y.1"] == list(range(0, 48, 4))

    def test_flatten_selected_features(self, batch, features):
        # create processor
        p = FlattenFeatures(
            FlattenFeaturesConfig(delimiter=".", to_flatten=["Y"])
        )
        p.prepare(features)

        # check features
        assert p.new_features == Features(
            {
                # flatten sequence
                "Y.0": Value("int32"),
                "Y.1": Value("int32"),
                "Y.2": Value("int32"),
            }
        )

        # apply processor
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        assert batch["Y.0"] == list(range(1, 13))
        assert batch["Y.1"] == list(range(2, 14))
        assert batch["Y.2"] == list(range(3, 15))

    def test_feature_key_not_found(self, features):
        # create processor
        p = FlattenFeatures(FlattenFeaturesConfig(to_flatten=["INVALID_KEY"]))
        # should raise key error as key is not present in features
        with pytest.raises(KeyError):
            p.prepare(features)

    def test_warning_when_overwriting_mapping(self, features):
        # create processor
        p = FlattenFeatures(FlattenFeaturesConfig(mapping={"new_A": "A"}))
        # should raise key error as key is not present in features
        with pytest.warns(UserWarning):
            p.prepare(features)
