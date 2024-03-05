import datetime
from dataclasses import dataclass, field
from typing import Annotated, Literal

import datasets
import pyarrow as pa
import pydantic
from datasets.packaged_modules.json.json import Json, JsonConfig

# map datasets value dtype to
DATASETS_VALUE_TYPE_MAPPING = {
    "bool": bool,
    "int8": int,
    "int16": int,
    "int32": int,
    "int64": int,
    "uint8": int,
    "uint16": int,
    "uint32": int,
    "uint64": int,
    "float16": float,
    "float32": float,
    "float64": float,
    "string": str,
    "large_string": str,
    "date32": datetime.datetime,
    "date64": datetime.datetime,
    "time32": datetime.time,
    "time64": datetime.time,
}


def pydantic_model_from_features(
    features: datasets.Features,
) -> pydantic.BaseModel:
    """Create a pydantic model from dataset features

    Arguments:
        features (Features): datasets features to build the pydantic model for

    Returns:
        model (pydantic.BaseModel):
            pydantic model matching the structure of the dataset features.
    """

    fields = {}
    for k, field_type in features.items():
        if isinstance(field_type, datasets.Value):
            # get data type for the given field
            dtype = DATASETS_VALUE_TYPE_MAPPING.get(
                field_type.dtype, field_type.pa_type.to_pandas_dtype()
            )
            # set field
            fields[k] = (dtype | None, None)

        elif isinstance(field_type, datasets.ClassLabel):
            fields[k] = (Literal[*field_type.names] | None, None)

        elif isinstance(field_type, datasets.Sequence):
            # infer dtype for sequence values
            dtype = (
                pydantic_model_from_features({"field": field_type.feature})
                .model_fields["field"]
                .annotation
            )
            # set field
            fields[k] = (
                Annotated[
                    list[dtype],
                    pydantic.BeforeValidator(
                        lambda v: v if v is not None else []
                    ),
                ],
                pydantic.Field(default_factory=list, validate_default=True),
            )

        elif isinstance(field_type, (dict, datasets.Features)):
            model = pydantic_model_from_features(field_type)
            # set field
            fields[k] = (
                Annotated[
                    model,
                    pydantic.BeforeValidator(
                        lambda v: v if v is not None else model()
                    ),
                ],
                pydantic.Field(default_factory=model, validate_default=True),
            )

    return pydantic.create_model(
        "Model",
        **fields,
        __config__=pydantic.ConfigDict(
            arbitrary_types_allowed=True, validate_assignment=True
        ),
    )


@dataclass
class TypedJsonDatasetConfig(JsonConfig):
    """Typed Json Dataset Configuration

    Matches the huggingface datasets json dataset implementation.
    Please refer to the huggingface documentation for more information.

    The attributes of the configuration are typically set by providing
    them as keyword arguments to the `datasets.load_dataset` function.

    Attributes:
        data_files (str | list[str] | dict[str,str|list[str]):
            files to load
        features (datasets.Features):
            dataset features, required for type checking
        **kwargs (Any):
            please refer to huggingface documentation
    """

    # features are required and not
    # optional as in the base json cofig
    features: datasets.Features = None

    _feature_model: pydantic.BaseModel = field(init=False)
    _batch_feature_model: pydantic.BaseModel = field(init=False)

    def __post_init__(self) -> None:
        if self.features is None:
            raise ValueError(
                "No dataset features provided. Please specify the expeted "
                "dataset features for type checking."
            )
        # create pydantic feature model
        self._feature_model = pydantic_model_from_features(self.features)
        self._batch_feature_model = pydantic.create_model(
            "BatchModel", data=(list[self._feature_model], ...)
        )


class TypedJsonDataset(Json):
    """Typed Json Dataset

    Typically used by call to `datasets.load_dataset with appropriate
    keyword arguments (see `TypedJsonDatasetConfig` for defails)

    ```
    datasets.load_dataset('hyped.data.io.datasets.typed_json', **kwargs)
    ```
    """

    BUILDER_CONFIG_CLASS = TypedJsonDatasetConfig

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        # validate table using pydantic
        data = {"data": pa_table.to_pylist()}
        data = self.config._batch_feature_model.model_validate(data)
        return pa.Table.from_pylist(data.model_dump()["data"])
