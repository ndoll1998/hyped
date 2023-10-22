from hyped.data.processors.features.format import (
    FormatFeaturesConfig,
    FormatFeatures,
)
from datasets import Features, Value, Sequence
import pytest


class TestFormatWithFlatFeatures:
    @pytest.fixture()
    def batch(self):
        return {
            "X": list(range(0, 12)),
            "Y": list(range(12, 24)),
            "A": list(map(chr, range(0, 12))),
        }

    @pytest.fixture()
    def features(self):
        return Features(
            {"X": Value("int32"), "Y": Value("int32"), "A": Value("string")}
        )

    def test_rename(self, batch, features):
        # create rename processor
        p = FormatFeatures(
            FormatFeaturesConfig(mapping={"new_X": "X", "new_Y": "Y"})
        )

        # prepare
        p.prepare(features)
        # check features
        assert p.new_features["new_X"] == features["X"]
        assert p.new_features["new_Y"] == features["Y"]

        # apply
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        assert batch["X"] == batch["new_X"]
        assert batch["Y"] == batch["new_Y"]

    def test_pack_in_list(self, batch, features):
        # create processor
        p = FormatFeatures(FormatFeaturesConfig(mapping={"XY": ["X", "Y"]}))

        # prepare and check features
        p.prepare(features)
        assert p.new_features["XY"].feature == Value("int32")
        assert p.new_features["XY"].length == 2

        # apply and check content
        batch = p.batch_process(batch, index=range(12), rank=0)
        assert batch["XY"] == list(map(list, zip(batch["X"], batch["Y"])))

    def test_error_on_pack_different_types_in_list(self, batch, features):
        # create processor
        p = FormatFeatures(FormatFeaturesConfig(mapping={"Z": ["X", "A"]}))

        # prepare and check features
        with pytest.raises(TypeError):
            p.prepare(features)

    def test_pack_in_dict(self, batch, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={"XYA": {"X": "X", "YA": {"Y": "Y", "A": "A"}}}
            )
        )

        # prepare
        p.prepare(features)
        # check features
        assert p.new_features == Features(
            {
                "XYA": {
                    "X": features["X"],
                    "YA": {"Y": features["Y"], "A": features["A"]},
                }
            }
        )

        # apply
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        for i, (xya, x, y, a) in enumerate(
            zip(batch["XYA"], batch["X"], batch["Y"], batch["A"])
        ):
            assert xya == {"X": x, "YA": {"Y": y, "A": a}}

    def test_pack_in_list_of_dicts(self, batch, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "XYA": [{"XorY": "X", "A": "A"}, {"XorY": "Y", "A": "A"}]
                }
            )
        )

        # prepare
        p.prepare(features)
        assert p.new_features == Features(
            {
                "XYA": Sequence(
                    {"XorY": Value("int32"), "A": Value("string")}, length=2
                )
            }
        )

        # apply
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        for i, (xya, x, y, a) in enumerate(
            zip(batch["XYA"], batch["X"], batch["Y"], batch["A"])
        ):
            assert xya == [{"XorY": x, "A": a}, {"XorY": y, "A": a}]


class TestFormatWithNestedFeatures:
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

    def test_unexpected_key_type(self, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "Y.0": ("Y", 1.2),
                }
            )
        )

        # should raise type error as float is an invalid key type
        with pytest.raises(TypeError):
            p.prepare(features)

    def test_access_sequence(self, batch, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "Y.0": ("Y", 0),
                    "Y.1": ("Y", 1),
                    "Y.2": ("Y", 2),
                }
            )
        )

        # prepare
        p.prepare(features)
        # check features
        assert p.new_features["new_X"] == Value("int32")
        assert p.new_features["Y.0"] == Value("int32")
        assert p.new_features["Y.1"] == Value("int32")
        assert p.new_features["Y.2"] == Value("int32")

        # apply
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        assert batch["X"] == batch["new_X"]
        assert batch["Y.0"] == [x for x, _, _ in batch["Y"]]
        assert batch["Y.1"] == [x for _, x, _ in batch["Y"]]
        assert batch["Y.2"] == [x for _, _, x in batch["Y"]]

    def test_sequence_index_out_of_range_error(self, features):
        # create processor
        p = FormatFeatures(
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

        # should raise index error as sequence y has length 3
        with pytest.raises(IndexError):
            p.prepare(features)

    def test_sequence_invalid_key(self, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(mapping={"new_X": "X", "Y.x": ("Y", "x")})
        )

        # trying to index a sequence with string key
        with pytest.raises(TypeError):
            p.prepare(features)

    def test_access_mapping(self, batch, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={"new_X": "X", "A.x": ("A", "x"), "A.y": ("A", "y")}
            )
        )

        # prepare
        p.prepare(features)
        # check features
        assert p.new_features["new_X"] == Value("int32")
        assert p.new_features["A.x"] == Value("int32")
        assert p.new_features["A.y"] == Sequence(Value("int32"), length=2)

        # apply
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        assert batch["X"] == batch["new_X"]
        assert batch["A.x"] == [x["x"] for x in batch["A"]]
        assert batch["A.y"] == [x["y"] for x in batch["A"]]

    def test_mapping_key_not_found(self, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "A.x": ("A", "x"),
                    "A.y": ("A", "y"),
                    "A.z": ("A", "z"),
                }
            )
        )

        # should raise key error as A has no key z
        with pytest.raises(KeyError):
            p.prepare(features)

    def test_mapping_invalid_key(self, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(mapping={"new_X": "X", "A.0": ("A", 0)})
        )

        # trying to index a mapping with integer key
        with pytest.raises(TypeError):
            p.prepare(features)

    def test_access_nested(self, batch, features):
        # create processor
        p = FormatFeatures(
            FormatFeaturesConfig(
                mapping={
                    "new_X": "X",
                    "A.x": ("A", "x"),
                    "A.y.0": ("A", "y", 0),
                    "A.y.1": ("A", "y", 1),
                }
            )
        )

        # prepare
        p.prepare(features)
        # check features
        assert p.new_features["new_X"] == Value("int32")
        assert p.new_features["A.x"] == Value("int32")
        assert p.new_features["A.y.0"] == Value("int32")
        assert p.new_features["A.y.1"] == Value("int32")

        # apply
        batch = p.batch_process(batch, index=range(12), rank=0)
        # check content
        assert batch["X"] == batch["new_X"]
        assert batch["A.x"] == [x["x"] for x in batch["A"]]
        assert batch["A.y.0"] == [x["y"][0] for x in batch["A"]]
        assert batch["A.y.1"] == [x["y"][1] for x in batch["A"]]
