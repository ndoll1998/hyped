from inspect import isclass
from datasets import Features, Sequence, Value
from datasets.features.features import FeatureType

INT_TYPES = [
    Value("int8"),
    Value("int16"),
    Value("int32"),
    Value("int64"),
    Value("int64"),
]

UINT_TYPES = [
    Value("uint8"),
    Value("uint16"),
    Value("uint32"),
    Value("uint64"),
    Value("uint64"),
]

FLOAT_TYPES = [
    Value("float16"),
    Value("float32"),
    Value("float64"),
]

INDEX_TYPES = INT_TYPES + UINT_TYPES


def check_feature_exists(name: str, features: Features) -> bool:
    """Check if feature exists in feature mapping

    Raises KeyError if feature is not present.

    Arguments:
        name (str): name of the feature to check for
        features (Features): feature mapping to check

    Returns:
        exists (bool): whether the feature exists
    """
    return name in features


def check_feature_equals(
    feature: FeatureType, target: FeatureType | list[FeatureType]
) -> bool:
    """Check whether a given feature equals a target feature

    This confirms exact matches, including for instance
    checking that the lengths of sequences match.

    Arguments:
        feature (FeatureType):
            the feature to check, for mapping the function is
            called recursive for each member
        target (FeatureType | list[FeatureType]):
            the target feature or list of valid target features.
            If the target is a list or tuple of length one that
            it is still considered to be a feature and not a list
            of features as this is also a definition of a sequence
            feature type.

    Returns:
        is_equal (bool):
            bool indicating whether the feature matches
            the target feature type
    """

    if isinstance(target, (list, tuple)):
        # if multiple valid targets are given the feature
        # should match any one of them
        if (len(target) > 1) or (len(target) == 0):
            return any(check_feature_equals(feature, t) for t in target)
        # a list of length one is a valid definition of sequence
        return check_feature_is_sequence(
            feature, target[0]
        ) and check_sequence_lengths_match(feature, target)

    # a list of length one is a valid definition of a sequence
    if isinstance(feature, list):
        # at this point the target cannot be a list so to
        # align with the given feature is must be sequence
        if isinstance(target, Sequence):
            return check_feature_is_sequence(
                feature, target.feature
            ) and check_sequence_lengths_match(feature, target)

        return target is Sequence

    if isinstance(feature, dict):
        # make sure the target is also a mapping
        # with the same keys and ensure that the
        # features are equal as well
        return (
            isinstance(target, dict)
            and (feature.keys() == target.keys())
            and all(
                check_feature_equals(feature[k], target[k])
                for k in feature.keys()
            )
        )

    if isclass(target):
        # if only the target feature class and
        # not the exact target feature is specified
        return isinstance(feature, target)

    # otherwise it should just match the target
    return feature == target


def check_feature_is_sequence(
    feature: FeatureType,
    value_type: None | FeatureType | list[FeatureType] = None,
) -> bool:
    """Check if a given feature is a sequence of values of
    a given value type (and arbitrary length).

    Arguments:
        feature (FeatureType):
            the feature to check
        value_type (None | FeatureType | list[FeatureType]):
            the expected sequence value type or a target
            of valid value types. If the value type is a list or
            tuple of length one, it is still considered to be a
            feature and not a list of features as this is also a
            valid definition of a sequence feature type. If the
            value type is set to None, the item type is not
            checked.

    Returns:
        is_sequence (bool):
            whether the feature is a sequence of (one of) the
            given value type(s)
    """

    if value_type is None:
        # only check if the feature is a sequence
        return isinstance(feature, (Sequence, list, tuple))

    return (
        # either its a sequence, then check the value type
        isinstance(feature, Sequence)
        and check_feature_equals(feature.feature, value_type)
    ) or (
        # or its a list, then also check the value type
        isinstance(feature, (list, tuple))
        and check_feature_equals(feature[0], value_type)
    )


def get_sequence_length(seq: Sequence | list | tuple) -> int:
    """Get the length of a given sequence feature

    Arguments:
        seq (Sequence | list | tuple): sequence to get the length of

    Returns:
        length (int):
            the length of the given sequence. Returns -1 for
            sequences of undefined length
    """
    assert isinstance(seq, (Sequence, list, tuple))
    return seq.length if isinstance(seq, Sequence) else -1


def check_sequence_lengths_match(
    seq_A: Sequence | list | tuple, seq_B: Sequence | list | tuple
) -> bool:
    """Check whether the lengths of two sequences match

    Arguments:
        sequence_A (Sequence | list | tuple): sequence A
        sequence_B (Sequence | list | tuple): sequence B

    Returns:
        match (bool): bool indicating if the lengths match
    """
    return get_sequence_length(seq_A) == get_sequence_length(seq_B)


def raise_feature_exists(name: str, features: Features) -> None:
    """Check if feature exists in feature mapping

    Arguments:
        name (str): name of the feature to check for
        features (Features): feature mapping to check

    Raises:
        exp (KeyError): when the feature doesn't exist
    """
    if not check_feature_exists(name, features):
        raise KeyError(
            "`%s` not present in features, valid feature keys are %s"
            % (name, list(features.keys()))
        )


def raise_feature_equals(
    name: str, feature: FeatureType, target: FeatureType | list[FeatureType]
) -> None:
    """Check whether a given feature equals a target feature

    This confirms exact matches, including for instance
    checking that the lengths of sequences match.

    Arguments:
        name (str):
            the name of the feature, only used in the error message
        feature (FeatureType):
            the feature to check
        target (FeatureType | list[FeatureType]):
            the target feature or list of valid target features.
            If the target is a list or tuple of length one that
            it is still considered to be a feature and not a list
            of features as this is also a definition of a sequence
            feature type.

    Raises:
        exc (TypeError): when the feature doesn't equal the target
    """

    if not check_feature_equals(feature, target):
        if isinstance(target, list):
            # slightly different error message for list of
            # value feature types
            raise TypeError(
                "Expected `%s` to be a one of the types "
                "in %s, got %s" % (name, target, feature)
            )

        raise TypeError(
            "Expected `%s` to be of type %s, got %s" % (name, target, feature)
        )


def raise_features_align(
    name_A: str, name_B: str, feature_A: FeatureType, feature_B: FeatureType
) -> None:
    """Check if two features align/match

    Arguments:
        name_A (str): name of feature A, only used in error message
        name_B (str): name of feature B, only used in error message
        feature_A (FeatureType): feature A to compare
        feature_B (FeatureType): feature B to compare

    Raises:
        exp (TypeError): when features don't match
    """

    if not check_feature_equals(feature_A, feature_B):
        raise TypeError(
            "Feature type of %s doesn't match the feature type of %s,"
            " got %s != %s" % (name_A, name_B, feature_A, feature_B)
        )


def raise_feature_is_sequence(
    name: str,
    feature: FeatureType,
    value_type: None | FeatureType | list[FeatureType] = None,
) -> None:
    """Check if a given feature is a sequence of values of
    a given value type (and arbitrary length).

    Arguments:
        name (str):
            the name of the feature, only used in the error message
        feature (FeatureType):
            the feature to check
        value_type (FeatureType | list[FeatureType]):
            the expected sequence value type or a target
            of valid value types. If the value type is a list or
            tuple of length one, it is still considered to be a
            feature and not a list of features as this is also a
            valid definition of a sequence feature type. If the
            value type is set to None, the item type is not
            checked.

    Raises:
        exc (TypeError):
            when the value type doesn't match the expected value type
    """
    if not check_feature_is_sequence(feature, value_type):
        if value_type is None:
            raise TypeError(
                "Expected `%s` to be a sequence, got %s" % (name, feature)
            )

        if isinstance(value_type, (list, tuple)) and (len(value_type) > 1):
            # slightly different error message for list of
            # value feature types
            raise TypeError(
                "Expected `%s` to be a sequence of one of the types "
                "in %s, got %s" % (name, value_type, feature)
            )

        raise TypeError(
            "Expected `%s` to be a sequence of type %s, got %s"
            % (name, value_type, feature)
        )
