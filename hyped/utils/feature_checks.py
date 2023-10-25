from datasets import Features, Sequence
from datasets.features.features import FeatureType


def check_feature_exists(name: str, features: Features) -> None:
    """Check if feature exists in feature mapping

    Raises KeyError if feature is not present.

    Arguments:
        name (str): name of the feature to check for
        features (Features): feature mapping to check
    """
    if name not in features:
        raise KeyError("`%s` not present in features!" % name)


def check_feature_is_sequence(
    name: str, feature: FeatureType, value: FeatureType
) -> None:
    """Check if a given feature is a sequence of values of
    a given value type. This function ignore the sequence length

    Raises TypeError if feature type does not match.

    Arguments:
        name (str): the name of the feature, only used in the error message
        feature (FeatureType): the feature to check
        value (FeatureType): the expected value type
    """
    if not (isinstance(feature, Sequence) and feature.feature == value):
        raise TypeError(
            "Expected `%s` to be a sequence of integers, "
            "got %s" % (name, feature)
        )
