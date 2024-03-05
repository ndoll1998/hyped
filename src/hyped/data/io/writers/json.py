import os
from typing import Any

import orjson
from torch.utils.data._utils.worker import get_worker_info

from .base import BaseDatasetConsumer


class JsonDatasetWriter(BaseDatasetConsumer):
    """Json Dataset Writer

    Implements the `BaseDatasetConsumer` class to write a dataset
    to the disk in json-line format.

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

    def __init__(
        self, save_dir: str, exist_ok: bool = False, **kwargs
    ) -> None:
        super(JsonDatasetWriter, self).__init__(**kwargs)
        self.save_dir = save_dir
        # create output directory
        os.makedirs(self.save_dir, exist_ok=exist_ok)

    def initialize_worker(self) -> None:
        """Open the save file for the worker"""

        worker_info = get_worker_info()
        # create file paths
        worker_info.args.save_file_path = os.path.join(
            self.save_dir, "data_shard_%i.jsonl" % worker_info.id
        )
        worker_info.args.features_file_path = os.path.join(
            self.save_dir, "features.json"
        )
        # open data save file
        worker_info.args.save_file = open(
            worker_info.args.save_file_path, "wb+"
        )

        if worker_info.id == 0:
            # save the datasets features
            with open(worker_info.args.features_file_path, "wb+") as f:
                f.write(orjson.dumps(worker_info.dataset.features.to_dict()))

    def finalize_worker(self) -> None:
        """Cleanup and close the save file"""

        worker_info = get_worker_info()
        # check if the file is empty
        worker_info.args.save_file.seek(0, os.SEEK_END)
        is_empty = worker_info.args.save_file.tell() == 0
        # remove last character in current file
        # which is an unnecessary newline character
        if not is_empty:
            worker_info.args.save_file.seek(
                worker_info.args.save_file.tell() - 1, os.SEEK_SET
            )
            worker_info.args.save_file.truncate()
        # close the file
        worker_info.args.save_file.close()

        # delete the file if it is empty
        if is_empty:
            os.remove(worker_info.args.save_file_path)

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
        worker_info.args.save_file.write(orjson.dumps(example) + b"\n")
