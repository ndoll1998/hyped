import os
from datetime import datetime
from typing import Any

import datasets
import orjson
import pytest
from datasets import ClassLabel, Features, Sequence, Value

# register custom datasets
import hyped.data.io.datasets  # noqa: F401


class TestTypedJsonDataset(object):
    @pytest.fixture
    def features(self) -> Features:
        return Features(
            {
                "int": Value("int32"),
                "string": Value("string"),
                "class": ClassLabel(names=list("ABC")),
                "sequence": Sequence(Value("int32")),
                "date": Value("date32"),
                "mapping": Features(
                    {"a": Value("int32"), "b": Value("int32")}
                ),
            }
        )

    @pytest.fixture
    def data(self) -> list[dict[str, Any]]:
        return [
            {
                "int": n,
                "string": "string-w\n-%i" % n,
                "class": "ABC"[n % 3],
                "sequence": [n, 2 * n],
                "date": datetime(2010, 1 + n % 12, 1),
                "mapping": {"a": n, "b": 2 * n},
            }
            for n in range(10)
        ]

    @pytest.fixture
    def data_file(self, data, tmp_path) -> str:
        data_file = os.path.join(tmp_path, "file.jsonl")
        # create sample data that matches the features
        with open(data_file, "wb+") as f:
            f.write(b"\n".join(map(orjson.dumps, data)))

        return data_file

    def test_loading(self, features, data_file):
        # load data using typed json dataset
        datasets.load_dataset(
            "hyped.data.io.datasets.typed_json",
            data_files=[data_file],
            features=features,
        )

    def test_type_error(self, features, data_file):
        # break feature type
        features["string"] = Value("int32")
        # string feature type shouldn't match
        with pytest.raises(datasets.exceptions.DatasetGenerationError):
            datasets.load_dataset(
                "hyped.data.io.datasets.typed_json",
                data_files=[data_file],
                features=features,
            )

    def test_fill_with_defaults(self, features, data_file):
        # add new feature
        features["new_feature"] = Value("string")
        # load data using typed json dataset
        ds = datasets.load_dataset(
            "hyped.data.io.datasets.typed_json",
            data_files=[data_file],
            features=features,
        )["train"]

        for item in ds:
            assert "new_feature" in item
            assert item["new_feature"] is None

    def test_partial_fill_with_defaults(self, features, data_file):
        partial_data = [{"int": n} for n in range(10)]

        with open(data_file, "ab") as f:
            f.write(b"\n")
            f.write(b"\n".join(map(orjson.dumps, partial_data)))

        # load data using typed json dataset
        datasets.load_dataset(
            "hyped.data.io.datasets.typed_json",
            data_files=[data_file],
            features=features,
        )["train"]

    def test_partial_fill_with_defaults_in_nested(self, features, data_file):
        partial_data = [{"int": n, "mapping": {"b": n}} for n in range(10)]

        with open(data_file, "ab") as f:
            f.write(b"\n")
            f.write(b"\n".join(map(orjson.dumps, partial_data)))

        # load data using typed json dataset
        datasets.load_dataset(
            "hyped.data.io.datasets.typed_json",
            data_files=[data_file],
            features=features,
        )["train"]

    def test_invalid_class_label(self, features, data_file):
        # remove class name from label space
        features["class"] = ClassLabel(names=list("AB"))
        # string feature type shouldn't match
        with pytest.raises(datasets.exceptions.DatasetGenerationError):
            datasets.load_dataset(
                "hyped.data.io.datasets.typed_json",
                data_files=[data_file],
                features=features,
            )
