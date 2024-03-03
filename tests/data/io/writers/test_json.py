import json
import os
from itertools import product

import datasets
import pytest

from hyped.data.io.writers.json import JsonDatasetWriter


class TestJsonDatasetWriter(object):
    @pytest.mark.parametrize("num_samples", [1024, 2048, 10000])
    @pytest.mark.parametrize(
        "num_proc, num_shards", product([2, 4, 8, 16], repeat=2)
    )
    def test_with_dummy_data(self, tmpdir, num_samples, num_proc, num_shards):
        # create dummy dataset
        ds = datasets.Dataset.from_dict({"id": range(num_samples)})
        ds = ds.to_iterable_dataset(num_shards=num_shards)
        # write dataset
        writer = JsonDatasetWriter(
            save_dir=tmpdir, exist_ok=True, num_proc=num_proc
        )
        writer.consume(ds)

        # load dataset features from disk
        with open(os.path.join(str(tmpdir), "features.json"), "r") as f:
            features = datasets.Features.from_dict(json.loads(f.read()))
        assert features == ds.features

        # load stored dataset
        stored_ds = datasets.load_dataset(
            "json", data_files="%s/*.jsonl" % str(tmpdir), features=features
        )
        stored_ds = stored_ds["train"].sort("id")
        # check dataset
        for i, item in enumerate(stored_ds):
            assert i == item["id"]

    @pytest.mark.parametrize("dataset", ["conll2003", "imdb"])
    def test_with_actual_data(self, tmpdir, dataset):
        # load dataset and add an index column to
        # recover the original order after saving
        ds = datasets.load_dataset(dataset, split="test")
        ds = ds.add_column("__index__", range(len(ds)))
        # shard it
        ds = ds.to_iterable_dataset(num_shards=8)
        # write dataset
        writer = JsonDatasetWriter(save_dir=tmpdir, exist_ok=True, num_proc=4)
        writer.consume(ds)

        # load dataset features from disk
        with open(os.path.join(str(tmpdir), "features.json"), "r") as f:
            features = datasets.Features.from_dict(json.loads(f.read()))
        assert features == ds.features

        # load stored dataset
        stored_ds = datasets.load_dataset(
            "json", data_files="%s/*.jsonl" % str(tmpdir), features=features
        )
        stored_ds = stored_ds["train"].sort("__index__")

        for original, stored in zip(ds, stored_ds):
            assert original == stored
