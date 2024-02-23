from itertools import chain
from typing import Any, Iterable, NewType

from datasets.features.features import Features, FeatureType, Sequence

from hyped.utils.feature_checks import (
    check_feature_equals,
    get_sequence_feature,
    get_sequence_length,
    raise_feature_equals,
    raise_feature_exists,
    raise_feature_is_sequence,
)

# TODO: write tests

# a feature key must either be a string or a path represented
# by a tuple of keys to follow in the feature mapping
FeatureKey = NewType("FeatureKey", str | tuple[str | int | slice])
# a feature collection is a collection of feature keys
# either in form of a list or in form of a dictionary
FeatureKeyCollection = (
    FeatureKey
    | list["FeatureKeyCollection"]
    | dict[str, "FeatureKeyCollection"]
)


def is_feature_key(key: FeatureKey) -> bool:
    """Test whether a given candidate matches the `FeatureKey` type.

    Arguments:
        key (FeatureKey): key candidate

    Returns:
        is_key (bool):
            boolean indicating whether the candidate matches the type
    """

    return isinstance(key, str) or (
        # or a path of only string keys
        isinstance(key, tuple)
        and all(isinstance(k, str | int | slice) for k in key)
    )


def is_simple_key(key: FeatureKey) -> bool:
    """Check whether a given feature key is simple.

    A feature key is considered simple if it only consists of
    strings.

    The concept behind a simple key is that is only
    indexes dictionaries, meaning it does not interact with
    sequences.

    It does not require fancy sequence slicing. Also modifying
    the value at the key modifies the full feature and not just
    a part of it.

    As a counter example consider the following complex keys

        - ("A", 0, "B")
        - ("A", slice(4, 10), "B")

    Both incorporate indexing a sequence feature and modifying it
    would only modify a specific value or sub-area of the sequence.

    Arguments:
        key (FeatureKey): key to check

    Returns:
        is_simple (bool): boolean indicating whether the key is simple
    """
    return (
        # either a simple string key
        isinstance(key, str)
        | (
            # or a path of only string keys
            isinstance(key, tuple)
            and all(isinstance(k, str) for k in key)
        )
    )


def key_cutoff_at_slice(key: FeatureKey) -> FeatureKey:
    """Cutoff given key at first occurance of a slice

    Consider the following example:

        ("A", "B", 0, "C", slice(-1), "D") => ("A", "B", 0)

    Arguments:
        key (FeatureKey): key to check

    Returns:
        cut_key (FeatureKey):
            truncated key guaranteed to not contain a slice
    """

    if is_simple_key(key):
        return key

    # find slice in key
    for i, k in enumerate(key):
        if isinstance(k, slice):
            # cutoff before slice
            return key[:i]

    # no slice detected
    return key


def raise_is_simple_key(key: FeatureKey):
    """Check whether a given feature key is simple.

    A feature key is considered simple if it only consists of
    strings.

    Arguments:
        key (FeatureKey): key to check

    Raises:
        exc (TypeError): when the given key is complex
    """

    if not is_simple_key(key):
        raise TypeError("Expected simple key, got %s" % str(key))


def iter_keys_in_features(
    features: FeatureType, max_depth: int = -1, max_seq_len_to_unpack: int = 8
) -> Iterable[FeatureKey]:
    """Iterate over all keys present in the given features

    Take for example the following feature mapping

        {
            "A": {"B": Value("string")},
            "X": Sequence(Value("int32"), length=2)
        }

    Then the iterator would yield the following keys

        ("A", "B"), ("X", 0), ("X", 1)

    Arguments:
        features (FeatureType): features to build the keys for
        max_depth (int):
            when set to a positive integer, the nested structure
            of the feature mapping will only be traversed to the
            specified depth. The maximum length of each key is
            restricted by this value. Defaults to -1.
        max_seq_len_to_unpack (int):
            upper threshold of length to flatten sequences. If the sequence
            length exceeds this threshold, the sequence will not be flattened

    Returns:
        keys (Iterable[FeatureKey]):
            iterator over present keys
    """

    if max_depth == 0:
        # trivial case, maximum depth reached
        yield tuple()

    elif isinstance(features, (dict, Features)):
        # recursivly flatten all features in mapping and
        # prefix each sub-key with the current key of the mapping
        yield from chain.from_iterable(
            (
                map((k,).__add__, iter_keys_in_features(v, max_depth - 1))
                for k, v in features.items()
            )
        )

    elif isinstance(features, Sequence):
        length = get_sequence_length(features)
        # only unpack sequences of fixed length
        if 0 < length < max_seq_len_to_unpack:
            yield from (
                (i,) + sub_key
                for sub_key in iter_keys_in_features(
                    get_sequence_feature(features), max_depth - 1
                )
                for i in range(length)
            )

        else:
            yield from map(
                (slice(-1),).__add__,
                iter_keys_in_features(
                    get_sequence_feature(features), max_depth - 1
                ),
            )

    else:
        # all other feature types are considered primitive/unpackable
        yield tuple()


def build_collection_from_keys(keys: list[FeatureKey]) -> FeatureKeyCollection:
    """Build a feature collection from a list of feature keys.

    The resulting feature collection matches the format of the
    feature keys, i.e. the feature keys apply to the collection.

    Take for example to following feature keys:

        [("A", "X"), ("A", "Y")]

    Then the resulting feature collection would be:

        {"A": {"X": ("A", "X"), "Y": ("A", "Y")}}

    Arguments:
        keys (list[FeatureKey]):
            list of feature keys to build a collection from

    Returns:
        collection (FeatureCollection):
            resulting feature collection
    """

    # TODO: support complex keys
    if not all(map(is_simple_key, keys)):
        raise NotImplementedError()

    # initialize feature key collection
    collection = {}

    # add all keys
    for key in keys:
        # currently only supports simple keys
        key = (key,) if isinstance(key, str) else key

        c = collection
        # follow key path in collection
        for k in key[:-1]:
            if k not in c:
                c[k] = {}
            c = c[k]
        # write key to collection
        c[key[-1]] = key

    return collection


def remove_feature(features: Features, key: FeatureKey) -> Features:
    """Remove a feature from a feature mapping

    Arguments:
        features (Features): features to remove the feature from
        key (FeatureKey): key to the feature to remove, must be a simple key

    Returns:
        remaining_features (Features): the remaining features
    """

    # can only remove simple features
    if not is_simple_key(key):
        raise ValueError(
            "Can only remove feature with simple key from features, "
            "got `%s`" % str(key)
        )
    # remove the feature at key from the given features
    key = (key,) if isinstance(key, str) else key
    get_feature_at_key(features, key[:-1]).pop(key[-1])
    # remove remaining features
    return features


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


def batch_get_value_at_key(
    examples: dict[str, list[Any]], key: FeatureKey
) -> Any:
    """Index a batch of examples with the given key and retrieve
    the batch of values.

    Arguments:
        example (dict[str, Any]): Batch of example to index.
        key (FeatureKey): The key or path specifying the value to extract.

    Returns:
        values (Any): the batch of values of the example at the given key.
    """

    if not isinstance(key, str) and (len(key) > 1):
        key = key[:1] + (slice(-1),) + key[1:]

    return get_value_at_key(examples, key)


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
