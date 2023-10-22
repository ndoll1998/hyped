from hyped.data.processors.features.filter import (
    FilterFeatures,
    FilterFeaturesConfig,
)
from datasets import Features, Sequence, Value
import pytest


class TestFilterFeatures:
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

    def test_keep_features(self, batch, features):
        # create processor
        p = FilterFeatures(FilterFeaturesConfig(keep=["Y", "A"]))

        # prepare and check features
        p.prepare(features)
        assert p.new_features == p.out_features
        assert p.new_features == Features(
            {
                "Y": Sequence(Value("int32"), length=3),
                "A": {
                    "x": Value("int32"),
                    "y": Sequence(Value("int32"), length=2),
                },
            }
        )
        # apply and check output
        out_batch = p.process(batch, index=range(12), rank=0)
        assert "X" not in out_batch
        assert out_batch["Y"] == batch["Y"]
        assert out_batch["A"] == batch["A"]

    def test_remove_features(self, batch, features):
        # create processor
        p = FilterFeatures(FilterFeaturesConfig(remove=["X"]))

        # prepare and check features
        p.prepare(features)
        assert p.new_features == p.out_features
        assert p.new_features == Features(
            {
                "Y": Sequence(Value("int32"), length=3),
                "A": {
                    "x": Value("int32"),
                    "y": Sequence(Value("int32"), length=2),
                },
            }
        )
        # apply and check output
        out_batch = p.process(batch, index=range(12), rank=0)
        assert "X" not in out_batch
        assert out_batch["Y"] == batch["Y"]
        assert out_batch["A"] == batch["A"]

    def test_keep_single_feature(self, batch, features):
        # create processor
        p = FilterFeatures(FilterFeaturesConfig(keep="Y"))

        # prepare and check features
        p.prepare(features)
        assert p.new_features == p.out_features
        assert p.new_features == Features(
            {
                "Y": Sequence(Value("int32"), length=3),
            }
        )
        # apply and check output
        out_batch = p.process(batch, index=range(12), rank=0)
        assert "X" not in out_batch
        assert "A" not in out_batch
        assert out_batch["Y"] == batch["Y"]

    def test_remove_single_features(self, batch, features):
        # create processor
        p = FilterFeatures(FilterFeaturesConfig(remove="X"))

        # prepare and check features
        p.prepare(features)
        assert p.new_features == p.out_features
        assert p.new_features == Features(
            {
                "Y": Sequence(Value("int32"), length=3),
                "A": {
                    "x": Value("int32"),
                    "y": Sequence(Value("int32"), length=2),
                },
            }
        )
        # apply and check output
        out_batch = p.process(batch, index=range(12), rank=0)
        assert "X" not in out_batch
        assert out_batch["Y"] == batch["Y"]
        assert out_batch["A"] == batch["A"]

    def test_error_on_invalid_key(self, features):
        # create processor
        p = FilterFeatures(FilterFeaturesConfig(keep="INVALID_KEY"))

        # should raise key error as keep key is invalid
        with pytest.raises(KeyError):
            p.prepare(features)

        # create processor
        p = FilterFeatures(FilterFeaturesConfig(remove="INVALID_KEY"))

        # should raise key error as remove key is invalid
        with pytest.raises(KeyError):
            p.prepare(features)

    def test_error_on_invalid_configuration(self, features):
        # create processor
        p = FilterFeatures(FilterFeaturesConfig())

        # should raise value error as no filter is specified
        with pytest.raises(ValueError):
            p.prepare(features)

        # create processor
        p = FilterFeatures(FilterFeaturesConfig(keep="A", remove="X"))

        # should raise key error as both filters are specified
        with pytest.raises(ValueError):
            p.prepare(features)
