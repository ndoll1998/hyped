import json
import os
from abc import ABC, abstractmethod

import datasets
import pytest


class BaseTestDatasetWriter(ABC):
    @pytest.fixture(params=[2, 4, 8, 16])
    def num_proc(self, request):
        return request.param

    @pytest.fixture(params=[2, 4, 8, 16])
    def num_shards(self, request):
        return request.param

    @pytest.fixture
    @abstractmethod
    def writer(self, tmpdir, num_proc):
        ...

    @abstractmethod
    def load_dataset(self, tmpdir, features):
        ...

    def assert_equal(self, a: dict, b: dict) -> bool:
        assert a == b

    @pytest.mark.parametrize("num_samples", [1024, 2048, 10000])
    def test_with_dummy_data(self, writer, tmpdir, num_samples, num_shards):
        # create dummy dataset
        ds = datasets.Dataset.from_dict({"id": range(num_samples)})
        ds = ds.to_iterable_dataset(num_shards=num_shards)
        # write dataset
        writer.consume(ds)
        # load dataset features from disk
        with open(os.path.join(str(tmpdir), "features.json"), "r") as f:
            features = datasets.Features.from_dict(json.loads(f.read()))
        assert features == ds.features
        # load stored dataset
        stored_ds = self.load_dataset(str(tmpdir), features)
        stored_ds = stored_ds["train"].sort("id")
        # check dataset
        for i, item in enumerate(stored_ds):
            self.assert_equal({"id": i}, item)

    @pytest.mark.parametrize("dataset", ["conll2003", "imdb"])
    def test_with_actual_data(self, writer, tmpdir, dataset):
        # load dataset and add an index column to
        # recover the original order after saving
        ds = datasets.load_dataset(dataset, split="test")
        ds = ds.add_column("__index__", range(len(ds)))
        # shard and write it
        ds = ds.to_iterable_dataset(num_shards=8)
        writer.consume(ds)
        # load dataset features from disk
        with open(os.path.join(str(tmpdir), "features.json"), "r") as f:
            features = datasets.Features.from_dict(json.loads(f.read()))
        assert features == ds.features
        # load stored dataset
        stored_ds = self.load_dataset(str(tmpdir), features)
        stored_ds = stored_ds["train"].sort("__index__")
        # check dataset
        for original, stored in zip(ds, stored_ds):
            self.assert_equal(original, stored)
