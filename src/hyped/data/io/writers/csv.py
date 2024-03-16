import csv
import os
from typing import Any

import _io
import datasets
from torch.utils.data._utils.worker import get_worker_info

from hyped.utils.feature_checks import check_feature_equals

from .base import BaseDatasetWriter


class CsvDatasetWriter(BaseDatasetWriter):
    """CSV Dataset Writer

    Implements the `BaseDatasetWriter` class to write a dataset
    to the disk in csv format.

    Arguments:
        save_dir (str): the directory to save the dataset in
        exist_ok (bool):
            whether it is ok to write to the directory if it
            already exists. Defaults to False.
        num_proc (None | int):
            The number of processes to use. Defaults to the number of
            cpu cores available.
        tqdm_kwargs (dict[str, Any]):
            extra keyword arguments passed to the tqdm progress bar
        tqdm_update_interval (float):
            the update interval in seconds in which the tqdm bar
            is updated
    """

    def worker_shard_file_obj(
        self, path: str, worker_id: int
    ) -> _io.TextIOWrapper:
        return open(os.path.join(path, "data_shard_%i.csv" % worker_id), "w+")

    def consume(
        self, data: datasets.Dataset | datasets.IterableDataset
    ) -> None:
        if not all(
            check_feature_equals(
                feature, (datasets.Value, datasets.ClassLabel)
            )
            for feature in data.features.values()
        ):
            # all features must be strings
            raise TypeError(
                "CSV Dataset Writer requires all dataset features to be "
                "primitives (i.e. instances of datasets.Value or "
                "datasets.ClassLabel), got %s" % str(data.features)
            )
        # consume dataset
        super(CsvDatasetWriter, self).consume(data)

    def initialize_worker(self) -> None:
        super(CsvDatasetWriter, self).initialize_worker()
        # create csv writer
        worker_info = get_worker_info()
        worker_info.args.csv_writer = csv.DictWriter(
            worker_info.args.save_file,
            fieldnames=list(worker_info.dataset.features.keys()),
        )
        # write header to file
        worker_info.args.csv_writer.writeheader()

    def consume_example(
        self,
        shard_id: int,
        example_id: int,
        example: dict[str, Any],
    ) -> None:
        """Encode an example in json and write it to the worker's save file.

        Arguments:
            worker (mp.Process): worker process
            worker_id (int): worker id
            shard_id (int): dataset shard id
            example_id (int): example id in the current dataset shard
            example (dict[str, Any]): the example to consume
        """
        # save example to file in json format
        worker_info = get_worker_info()
        worker_info.args.csv_writer.writerow(example)
