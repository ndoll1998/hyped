import datasets
import pytest

from hyped.data.io.writers.json import JsonDatasetWriter
from tests.data.io.writers.base import BaseTestDatasetWriter


class TestJsonDatasetWriter(BaseTestDatasetWriter):
    @pytest.fixture
    def writer(self, tmpdir, num_proc):
        return JsonDatasetWriter(
            save_dir=tmpdir, exist_ok=True, num_proc=num_proc
        )

    def load_dataset(self, tmpdir, features):
        return datasets.load_dataset(
            "json", data_files="%s/*.jsonl" % tmpdir, features=features
        )
