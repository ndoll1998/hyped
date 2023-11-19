from hyped.utils.feature_checks import (
    raise_feature_is_sequence,
    raise_feature_exists,
    raise_feature_equals,
    check_feature_equals,
    get_sequence_length,
    get_sequence_feature,
)
from datasets.features.features import Features, FeatureType, Sequence
from typing import Any, Iterable

# TODO: write tests

# a feature key must either be a string or a path represented
# by a tuple of keys to follow in the feature mapping
FeatureKey = str | tuple[str | int | slice]
# a feature collection is a collection of feature keys
# either in form of a list or in form of a dictionary
FeatureKeyCollection = (
    FeatureKey
    | list["FeatureKeyCollection"]
    | dict[str, "FeatureKeyCollection"]
)


def get_feature_at_key(
    features: FeatureType, key: FeatureKey, _index: int = 0
) -> FeatureType:
    """Get the feature type of the feature indexed by the key.

    Arguments:
        features (FeatureType): The feature type to index with the given key.
        key (FeatureKey): The key or path specifying the feature to extract.
        _index (int):
            internal value used in recursion. Don't set this manually.

    Returns:
        feature (FeatureType):
            the feature type of the feature at the given key.
    """

    # trivial case, key is a string
    if isinstance(key, str):
        raise_feature_exists(key, features)
        return features[key]

    # trivial case, reached end of path
    if _index == len(key):
        return features

    cur_key = key[_index]
    cur_path = ("Path%s" % str(key[:_index])) if _index > 0 else "features"

    if isinstance(cur_key, (int, slice)):
        # check feature type must be sequence
        raise_feature_is_sequence(cur_path, features)
        # get sequence feature and length
        length = get_sequence_length(features)
        features = get_sequence_feature(features)
        # follow path
        features = get_feature_at_key(features, key, _index + 1)

        if isinstance(cur_key, int):
            # check integer key is in bounds
            if (cur_key >= 0) and (cur_key >= length):
                raise IndexError(
                    "Index `%i` out of bounds for sequence of "
                    "length `%i` of feature at path %s"
                    % (cur_key, length, cur_path)
                )
            # return the sequence item feature type
            return features

        if isinstance(cur_key, slice):
            # TODO: support more complex slices
            if cur_key != slice(-1):
                raise NotImplementedError(
                    "Currently only supports complete slicing "
                    "i.e. slice(0, -1, 1), got %s" % str(cur_key)
                )
            # pack feature in sequence
            return Sequence(features, length=length)

    elif isinstance(cur_key, str):
        # check feature type
        raise_feature_equals(cur_path, features, [Features, dict])

        # check key
        if cur_key not in features.keys():
            raise KeyError(
                "Key `%s` not present in feature mapping at "
                "path `%s`, valid keys are %s"
                % (cur_key, cur_path, list(features.keys()))
            )
        # recurse on feature at key
        return get_feature_at_key(features[cur_key], key, _index + 1)

    else:
        # unexpected key type in path
        raise TypeError(
            "Unexpected key type in path, valid types are `str`, "
            "`int` or `slice`, got %s" % cur_key
        )


def collect_features(
    features: Features, collection: FeatureKeyCollection
) -> FeatureType:
    """Collect all features requested in the feature collection
    and maintain the format of the collection

    Arguments:
        features (Features):
            source feature mapping from which to collect the requested features
        collection (FeatureKeyCollection):
            collection of feature keys to extract from the feature mapping

    Returns:
        collected_features (FeatureType):
            collected features in the format of the feature key collection
    """

    # check if the collection is a key or a path
    if isinstance(collection, (str, tuple)):
        return get_feature_at_key(features, key=collection)

    # check if the collection is a list of collections
    if isinstance(collection, list):
        # collect all features in specified in the list
        features = [
            collect_features(features, sub_collection)
            for sub_collection in collection
        ]
        # make sure the feature types match
        if not all(check_feature_equals(f, features[0]) for f in features[1:]):
            raise TypeError(
                "Expected all items of a sequence to be of the "
                "same feature type, got %s" % str(features)
            )
        # pack into sequence
        return Sequence(feature=features[0], length=len(features))

    # check if the collection is a dict of collections
    if isinstance(collection, dict):
        return Features(
            {
                key: collect_features(features, sub_collection)
                for key, sub_collection in collection.items()
            }
        )

    # unexpected type in mapping
    raise TypeError(
        "Unexpected feature key collection type, allowed types are "
        "(nested variants of) str, list and dict, got %s" % str(collection)
    )


def get_value_at_key(example: dict[str, Any], key: FeatureKey) -> Any:
    """Index the example with the given key and retrieve the value.

    Arguments:
        example (dict[str, Any]): The example to index.
        key (FeatureKey): The key or path specifying the value to extract.

    Returns:
        value (Any): the value of the example at the given key.
    """

    # trivial case, key is a string
    if isinstance(key, str):
        return example[key]

    # trivial case, empty path
    if len(key) == 0:
        return example

    cur_key = key[0]

    if isinstance(cur_key, slice):
        # recurse further for each item in the sequence
        return [get_value_at_key(value, key[1:]) for value in example]

    # works for both sequences and mappings
    return get_value_at_key(example[cur_key], key[1:])


def collect_values(
    example: dict[str, Any], collection: FeatureKeyCollection
) -> Any:
    """Collect all values requested in the feature collection
    and maintain the format of the collection

    Arguments:
        example (dict[str, Any]):
            example from which to collect the requested values
        collection (FeatureKeyCollection):
            collection of feature keys to extract from the example

    Returns:
        collected_values (Any):
            collected values in the format of the feature key collection
    """

    # check if the collection is a key or a path
    if isinstance(collection, (str, tuple)):
        return get_value_at_key(example, key=collection)

    # check if the collection is a list of collections
    if isinstance(collection, list):
        return [
            collect_values(example, sub_collection)
            for sub_collection in collection
        ]

    # check if the collection is a dict of collections
    if isinstance(collection, dict):
        return {
            key: collect_values(example, sub_collection)
            for key, sub_collection in collection.items()
        }

    # unexpected type in mapping
    raise TypeError(
        "Unexpected feature key collection type, allowed types are "
        "(nested variants of) str, list and dict, got %s" % str(collection)
    )


def iter_batch(batch: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    """Iterator over all examples in the given batch

    Arguments:
        batch (dict[str, list[Any]]): batch to iterate over

    Yields:
        example (dict[str, Any]): single example of the batch
    """
    # trivial case, empty batch
    if len(batch) == 0:
        return
    # infer batch size from given batch
    batch_size = len(next(iter(batch.values())))
    # yield each example in the batch
    for i in range(batch_size):
        yield {key: values[i] for key, values in batch.items()}
