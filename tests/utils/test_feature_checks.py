from hyped.utils.feature_checks import (
    check_feature_exists,
    check_feature_equals,
    check_feature_is_sequence,
    check_sequence_lengths_match,
    check_object_matches_feature,
    get_sequence_length,
    get_sequence_feature,
    raise_feature_exists,
    raise_feature_equals,
    raise_feature_is_sequence,
    raise_object_matches_feature,
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
            # restrict feature class only
            [Value("int32"), Value],
            [Value("int64"), Value],
            [Value("string"), Value],
            [[Value("int32")], Sequence],
            [Sequence(Value("string")), Sequence],
            [Sequence(Value("string")), [Value]],
            [Value("string"), [Sequence, Value]],
            [
                Features({"A": Value("int32"), "B": Value("int32")}),
                Features({"A": Value, "B": Value("int32")}),
            ],
            [
                Features({"A": Value("int32"), "B": Value("int32")}),
                Features({"A": Value, "B": Value}),
            ],
            [
                Features({"A": Value("int32"), "B": Sequence(Value("int32"))}),
                Features({"A": Value, "B": Sequence}),
            ],
            [
                Features({"A": Value("int32"), "B": Sequence(Value("int32"))}),
                Features({"A": Value, "B": [Value]}),
            ],
            [
                Features({"A": Value("int32"), "B": Sequence(Value("int32"))}),
                [
                    Features({"A": [Value], "B": [Value]}),
                    Features({"A": Value, "B": [Value]}),
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
            # restrict feature class only
            [Value("int32"), Sequence],
            [Value("int64"), Sequence],
            [Value("string"), [Sequence, Features]],
            [
                Features({"A": Value("int32"), "B": Value("int32")}),
                Features({"A": Value, "B": Value("int64")}),
            ],
            [
                Features({"A": Value("int32"), "B": Value("int32")}),
                Features({"A": Value, "B": Sequence}),
            ],
            [
                Features({"A": Value("int32"), "B": Sequence(Value("int32"))}),
                Features({"A": Value, "B": Value}),
            ],
            [
                Features({"A": Value("int32"), "B": Sequence(Value("int32"))}),
                Features({"A": Value, "B": [Sequence]}),
            ],
            [
                Features({"A": Value("int32"), "B": Sequence(Value("int32"))}),
                [
                    Features({"A": [Value], "B": [Value]}),
                    Features({"A": Value, "B": Value}),
                ],
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
            [[Value("int32")], None],
            [Sequence(Value("int32")), None],
            [[Value("int64")], None],
            [Sequence(Value("int64")), None],
            [[Value("string")], None],
            [Sequence(Value("string")), None],
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
            [[Value("int32")], Value],
            [Sequence(Value("int32")), Value],
            [Sequence(Value("int32"), length=2), Value],
            [Sequence(Value("int32")), [Value, Sequence]],
            [Sequence([Value("int32")]), [Value]],
            [Sequence([Value("int32")]), Sequence],
            [
                Sequence(Sequence(Value("int32"))),
                Sequence,
            ],
            [
                Sequence(Sequence(Value("int32")), length=2),
                Sequence,
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


class TestGetSequenceLength:
    @pytest.mark.parametrize(
        "seq, length",
        [
            [[Value("int32")], -1],
            [Sequence(Value("int32")), -1],
            [Sequence(Value("int32"), length=8), 8],
            [Sequence(Value("int32"), length=16), 16],
            [Sequence(Value("int32"), length=32), 32],
        ],
    )
    def test(self, seq, length):
        assert get_sequence_length(seq) == length


class TestGetSequenceFeature:
    @pytest.mark.parametrize(
        "seq, feature",
        [
            [[Value("int32")], Value("int32")],
            [Sequence(Value("int32")), Value("int32")],
            [Sequence(Value("int32"), length=8), Value("int32")],
            [[Value("string")], Value("string")],
            [Sequence(Value("string")), Value("string")],
            [Sequence(Value("string"), length=8), Value("string")],
            [[Sequence(Value("string"))], Sequence(Value("string"))],
            [Sequence(Sequence(Value("string"))), Sequence(Value("string"))],
            [
                Sequence(Sequence(Value("string"), length=8)),
                Sequence(Value("string"), length=8),
            ],
        ],
    )
    def test(self, seq, feature):
        assert get_sequence_feature(seq) == feature


class TestObjectMatchesFeature:
    @pytest.mark.parametrize(
        "obj, feature",
        [
            [1, Value("int16")],
            [1, Value("int32")],
            [1, Value("int64")],
            ["test", Value("string")],
            [[1, 2, 3], Sequence(Value("int32"))],
            [[1, 2, 3], Sequence(Value("int32"), length=3)],
        ],
    )
    def test_true(self, obj, feature):
        # object should match the type
        assert check_object_matches_feature(obj, feature)
        # thus this shouldn't raise an error
        raise_object_matches_feature(obj, feature)

    @pytest.mark.parametrize(
        "obj, feature",
        [
            [1, Value("string")],
            ["test", Value("int32")],
            [[1, 2, 3], Sequence(Value("string"))],
            [[1, 2, 3], Sequence(Value("int32"), length=2)],
        ],
    )
    def test_false(self, obj, feature):
        # object doesn't match the type
        assert not check_object_matches_feature(obj, feature)
        # thus this should raise an error
        with pytest.raises(TypeError):
            raise_object_matches_feature(obj, feature)


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
