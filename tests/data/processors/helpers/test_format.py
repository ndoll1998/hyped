from hyped.data.processors.helpers.format import (
    FormatFeaturesConfig,
    FormatFeatures,
)
from datasets import Features, Value, Sequence
import pytest


@pytest.fixture()
def batch():
    return {
        "X": list(range(0, 12)),
        "Y": list(range(12, 24)),
        "A": list(map(chr, range(0, 12))),
    }


@pytest.fixture()
def features():
    return Features(
        {"X": Value("int32"), "Y": Value("int32"), "A": Value("string")}
    )


class TestFormat:
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
        # create rename processor
        p = FormatFeatures(FormatFeaturesConfig(mapping={"XY": ["X", "Y"]}))

        # prepare and check features
        p.prepare(features)
        assert p.new_features["XY"].feature == Value("int32")
        assert p.new_features["XY"].length == 2

        # apply and check content
        batch = p.batch_process(batch, index=range(12), rank=0)
        assert batch["XY"] == list(map(list, zip(batch["X"], batch["Y"])))

    def test_error_on_pack_different_types_in_list(self, batch, features):
        # create rename processor
        p = FormatFeatures(FormatFeaturesConfig(mapping={"Z": ["X", "A"]}))

        # prepare and check features
        with pytest.raises(TypeError):
            p.prepare(features)

    def test_pack_in_dict(self, batch, features):
        # create rename processor
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
        # create rename processor
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
