from hyped.utils.feature_checks import (
    check_feature_exists,
    check_feature_equals,
    check_feature_is_sequence,
    check_sequence_lengths_match,
    raise_feature_exists,
    raise_feature_equals,
    raise_feature_is_sequence,
)
from datasets import Features, Sequence, Value
import pytest


class TestFeatureExists:
    @pytest.fixture
    def features(self):
        return Features(
            {
                "A": Value("int32"),
                "B": Value("int32"),
            }
        )

    def test_exist(self, features):
        # exist
        assert check_feature_exists("A", features)
        assert check_feature_exists("B", features)
        # shouldn't raise error
        check_feature_exists("A", features)
        check_feature_exists("B", features)

    def test_doesnt_exist(self, features):
        # doesnt exist
        assert not check_feature_exists("X", features)
        assert not check_feature_exists("Y", features)
        # should raise error
        with pytest.raises(KeyError):
            raise_feature_exists("X", features)
        with pytest.raises(KeyError):
            raise_feature_exists("Y", features)


class TestFeatureEquals:
    @pytest.mark.parametrize(
        "feature,target",
        [
            # easy checks
            [Value("int32"), Value("int32")],
            [Value("string"), Value("string")],
            [Value("string"), [Value("int32"), Value("string")]],
            # compare sequences
            [Sequence(Value("int32")), Sequence(Value("int32"))],
            [
                Sequence(Value("int32"), length=2),
                Sequence(Value("int32"), length=2),
            ],
            [
                Sequence(Value("int32")),
                [Sequence(Value("string")), Sequence(Value("int32"))],
            ],
            [
                Sequence(Value("int32"), length=2),
                [Sequence(Value("int32")), Sequence(Value("int32"), length=2)],
            ],
            # implicit sequence definition
            [[Value("int32")], Sequence(Value("int32"))],
            [Sequence(Value("int32")), [Value("int32")]],
            [[Value("int32")], [[Value("string")], [Value("int32")]]],
            [[Value("int32")], [[Value("string")], Sequence(Value("int32"))]],
            # mappings
            [
                Features({"A": Value("int32"), "B": Value("int32")}),
                Features({"A": Value("int32"), "B": Value("int32")}),
            ],
            [
                Features({"A": Sequence(Value("int32")), "B": Value("int32")}),
                Features({"A": Sequence(Value("int32")), "B": Value("int32")}),
            ],
            [
                Features({"A": [Value("int32")], "B": Value("int32")}),
                Features({"A": Sequence(Value("int32")), "B": Value("int32")}),
            ],
            [
                Features(
                    {
                        "A": Sequence(Value("int32"), length=2),
                        "B": Value("int32"),
                    }
                ),
                [
                    Features(
                        {"A": Sequence(Value("int32")), "B": Value("int32")}
                    ),
                    Features(
                        {
                            "A": Sequence(Value("int32"), length=2),
                            "B": Value("int32"),
                        }
                    ),
                ],
            ],
            [
                Sequence(Features({"A": Value("int32"), "B": Value("int32")})),
                Sequence(Features({"A": Value("int32"), "B": Value("int32")})),
            ],
            [
                Sequence(
                    Features({"A": Value("int32"), "B": Value("int32")}),
                    length=2,
                ),
                Sequence(
                    Features({"A": Value("int32"), "B": Value("int32")}),
                    length=2,
                ),
            ],
            [
                [Features({"A": Value("int32"), "B": Value("int32")})],
                Sequence(Features({"A": Value("int32"), "B": Value("int32")})),
            ],
            [
                Sequence(Features({"A": Value("int32"), "B": Value("int32")})),
                [Features({"A": Value("int32"), "B": Value("int32")})],
            ],
            [
                Sequence(Features({"A": Value("int32"), "B": Value("int32")})),
                [
                    Sequence(
                        Features({"X": Value("int32"), "Y": Value("int32")})
                    ),
                    Sequence(
                        Features({"A": Value("int32"), "B": Value("int32")})
                    ),
                ],
            ],
            [
                Sequence(
                    Features({"A": Value("int32"), "B": Value("int32")}),
                    length=2,
                ),
                [
                    Sequence(
                        Features({"A": Value("int32"), "B": Value("int32")})
                    ),
                    Sequence(
                        Features({"A": Value("int32"), "B": Value("int32")}),
                        length=2,
                    ),
                ],
            ],
            # nested mappings
            [
                Features(
                    {
                        "A": {"X": Value("int32"), "Y": Value("string")},
                        "B": Value("int32"),
                    }
                ),
                Features(
                    {
                        "A": {"X": Value("int32"), "Y": Value("string")},
                        "B": Value("int32"),
                    }
                ),
            ],
            [
                Features(
                    {
                        "A": {
                            "X": Sequence(Value("int32")),
                            "Y": Value("string"),
                        },
                        "B": Value("int32"),
                    }
                ),
                Features(
                    {
                        "A": {"X": [Value("int32")], "Y": Value("string")},
                        "B": Value("int32"),
                    }
                ),
            ],
            [
                Features(
                    {
                        "A": {"X": Value("int32"), "Y": Value("string")},
                        "B": Value("int32"),
                    }
                ),
                [
                    Features(
                        {
                            "A": {"X": Value("int32"), "Z": Value("string")},
                            "B": Value("int32"),
                        }
                    ),
                    Features(
                        {
                            "A": {"X": Value("int32"), "Y": Value("string")},
                            "B": Value("int32"),
                        }
                    ),
                ],
            ],
            [
                Features(
                    {
                        "A": {
                            "X": Sequence(Value("int32"), length=2),
                            "Y": Value("string"),
                        },
                        "B": Value("int32"),
                    }
                ),
                [
                    Features(
                        {
                            "A": {"X": Value("int32"), "Y": Value("string")},
                            "B": Value("int32"),
                        }
                    ),
                    Features(
                        {
                            "A": {
                                "X": Sequence(Value("int32"), length=2),
                                "Y": Value("string"),
                            },
                            "B": Value("int32"),
                        }
                    ),
                ],
            ],
        ],
    )
    def test_is_equal(self, feature, target):
        # is equal
        assert check_feature_equals(feature, target)
        # and shouldn't raise an exception
        raise_feature_equals("name", feature, target)

    @pytest.mark.parametrize(
        "feature,target",
        [
            [Value("int32"), Value("string")],
            [Value("int32"), [Value("int64"), Value("string")]],
            [Sequence(Value("int32")), Sequence(Value("string"))],
            [[Value("int32")], [Value("string")]],
            [Sequence(Value("int32"), length=2), Sequence(Value("int32"))],
            [Sequence(Value("int32"), length=2), [Value("int32")]],
            [
                [Value("int32")],
                [
                    Sequence(Value("int32"), length=1),
                    Sequence(Value("int32"), length=2),
                ],
            ],
            [Features({"A": Value("int32")}), Features({"X": Value("int32")})],
            [
                Features({"A": Value("int32")}),
                [
                    Features({"A": Value("string")}),
                    Features({"A": Value("int64")}),
                ],
            ],
            [
                Sequence(Value("int32")),
                Features({"A": Sequence(Value("int32"))}),
            ],
        ],
    )
    def test_is_not_equal(self, feature, target):
        # should not equal
        assert not check_feature_equals(feature, target)
        # and should raise an exception
        with pytest.raises(TypeError):
            raise_feature_equals("name", feature, target)


class TestFeatureIsSequence:
    @pytest.mark.parametrize(
        "feature,value_type",
        [
            [[Value("int32")], Value("int32")],
            [Sequence(Value("int32")), Value("int32")],
            [Sequence(Value("int32"), length=2), Value("int32")],
            [Sequence(Value("int32")), [Value("string"), Value("int32")]],
            [Sequence([Value("int32")]), [Value("int32")]],
            [Sequence([Value("int32")]), Sequence(Value("int32"))],
            [
                Sequence(Sequence(Value("int32"))),
                Sequence(Value("int32")),
            ],
            [
                Sequence(Sequence(Value("int32")), length=2),
                Sequence(Value("int32")),
            ],
        ],
    )
    def test_is_sequence(self, feature, value_type):
        # is sequence
        assert check_feature_is_sequence(feature, value_type)
        # and shouldn't raise an error
        raise_feature_is_sequence("name", feature, value_type)

    @pytest.mark.parametrize(
        "feature,value_type",
        [
            [[Value("int32")], Value("string")],
            [Sequence(Value("int32")), Value("string")],
            [Sequence(Value("int32"), length=2), Value("string")],
            [Sequence(Value("int32")), [Value("string"), Value("int64")]],
            [Sequence([Value("int32")]), Value("int32")],
            [Sequence([Value("int32")]), [Value("string")]],
            [Sequence([Value("int32")]), Sequence(Value("string"))],
            [
                Sequence(Sequence(Value("int32"))),
                Value("string"),
            ],
            [
                Sequence(Sequence(Value("int32"))),
                Sequence(Value("string")),
            ],
            [
                Sequence(Sequence(Value("int32"))),
                Sequence(Value("int32"), length=2),
            ],
            [
                Sequence(Sequence(Value("int32")), length=2),
                Sequence(Value("string")),
            ],
            [
                Sequence(Sequence(Value("int32")), length=2),
                Sequence(Value("string"), length=2),
            ],
        ],
    )
    def test_is_not_sequence(self, feature, value_type):
        # is sequence
        assert not check_feature_is_sequence(feature, value_type)
        # and shouldn't raise an error
        with pytest.raises(TypeError):
            raise_feature_is_sequence("name", feature, value_type)


class TestSequenceLengthsMatch:
    @pytest.mark.parametrize(
        "A, B",
        [
            [Sequence(Value("int32")), Sequence(Value("int32"))],
            [
                Sequence(Value("int32"), length=4),
                Sequence(Value("int32"), length=4),
            ],
            [[Value("int32")], [Value("int32")]],
            [[Value("int32")], Sequence(Value("int32"))],
            [Sequence(Value("int32")), [Value("int32")]],
        ],
    )
    def test_lengths_match(self, A, B):
        assert check_sequence_lengths_match(A, B)

    @pytest.mark.parametrize(
        "A, B",
        [
            [
                Sequence(Value("int32"), length=2),
                Sequence(Value("int32"), length=4),
            ],
            [Sequence(Value("int32"), length=2), [Value("int32")]],
            [[Value("int32")], Sequence(Value("int32"), length=2)],
        ],
    )
    def test_lengths_dont_match(self, A, B):
        assert not check_sequence_lengths_match(A, B)
