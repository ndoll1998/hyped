import pytest
from datasets import Features, Sequence, Value

from hyped.utils.arrow import convert_features_to_arrow_schema
from hyped.utils.feature_checks import check_feature_equals


@pytest.mark.parametrize(
    "features",
    [
        # easy cases
        Features({"Test": Value("int32")}),
        Features({"Test": Sequence(Value("int32"), length=16)}),
        Features(
            {
                "A": Sequence(Value("int32"), length=16),
                "B": Sequence(Value("int32"), length=32),
            }
        ),
        Features(
            {
                "A": {
                    "0": Sequence(Value("int32"), length=16),
                    "1": Sequence(Value("int32"), length=16),
                },
                "B": Sequence(Value("int32"), length=32),
            }
        ),
        Features({"A": Sequence({"A": Value("int32"), "B": Value("int32")})}),
        Features(
            {
                "A": Sequence({"A": Value("int32"), "B": Value("int32")}),
                "B": Sequence(
                    {"A": Value("int32"), "B": Value("int32")}, length=16
                ),
                "C": Sequence(
                    {
                        "A": Value("int32"),
                        "B": Value("int32"),
                        "C": Value("string"),
                    }
                ),
            }
        ),
    ],
)
def test_convert_features_to_arrow_schema(features):
    # convert, reconstruct and check
    schema = convert_features_to_arrow_schema(features)
    assert check_feature_equals(features, Features.from_arrow_schema(schema))
