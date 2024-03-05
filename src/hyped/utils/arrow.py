import json

import pyarrow as pa
from datasets import Features, Sequence
from datasets.features.features import FeatureType


def get_nested_type(schema: FeatureType) -> pa.DataType:
    """converts a datasets.FeatureType into a pyarrow.DataType,
    and acts as the inverse of `datasets.Dataset.generate_from_arrow_type`.

    It performs double-duty as the implementation of Features.type and
    handles the conversion of datasets.Feature->pa.struct

    Source: https://github.com/huggingface/datasets/blob/1a598a0dfd699f7a7ebe9eb6273fb5ac4b9e519a/src/datasets/features/features.py#L1184

    Differs from the source in that it doesn't convert Sequence[dict]->dict[Sequence].
    Instead it follows the exact feature type provided.
    """  # noqa: E501
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, Features):
        # Features is subclass of dict, and dict order is
        # deterministic since Python 3.6
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})
    elif isinstance(schema, dict):
        # however don't sort on struct types since the order matters
        return pa.struct({key: get_nested_type(schema[key]) for key in schema})
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError(
                "When defining list feature, you should just "
                "provide one example of the inner type"
            )
        value_type = get_nested_type(schema[0])
        return pa.list_(value_type)
    elif isinstance(schema, Sequence):
        return pa.list_(get_nested_type(schema.feature), schema.length)

    # Other objects are callable which returns their data type
    # (ClassLabel, Array2D, Translation, Arrow datatype creation methods)
    return schema()


def convert_features_to_arrow_schema(features: Features) -> pa.Schema:
    """Convert dataset.Features to pyarrow.Schema

    This is similar to the `datasets.Dataset.arrow_schema` property
    but parses the features slightly differently to ensure the schema
    matches the features exactly (see `get_nested_type` for more information)

    Arguments:
        features (Features): dataset features instance to convert

    Returns:
        schema (pa.Schema): pyarrow schema representing dataset features
    """
    dtype = get_nested_type(features)
    hf_metadata = {"info": {"features": features.to_dict()}}
    return pa.schema(dtype).with_metadata(
        {"huggingface": json.dumps(hf_metadata)}
    )
