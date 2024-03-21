import datasets
import pytest

from hyped.data.io.writers.csv import CsvDatasetWriter
from tests.data.io.writers.base import BaseTestDatasetWriter


class TestCsvDatasetWriter(BaseTestDatasetWriter):
    @pytest.fixture
    def writer(self, tmpdir, num_proc):
        return CsvDatasetWriter(
            save_dir=tmpdir, exist_ok=True, num_proc=num_proc
        )

    def load_dataset(self, tmpdir, features):
        return datasets.load_dataset(
            "csv", data_files="%s/*.csv" % tmpdir, features=features
        )

    def test_with_actual_data(self, writer, tmpdir):
        super(TestCsvDatasetWriter, self).test_with_actual_data(
            writer, tmpdir, "imdb"
        )
