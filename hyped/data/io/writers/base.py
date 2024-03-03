import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from queue import Empty
from time import time
from typing import Any, Iterable

import datasets
from tqdm.auto import tqdm


class BaseDatasetConsumer(ABC):
    """Abstract base class for dataset consumers

    The idea of consumer classes is to stream the items of a given dataset
    in order to execute some function on them. This is especially usefull
    when saving a processed dataset that exceeds the memory requirements.

    The base class provides basic functionality as well as multiprocessing
    support to parallelize the workload.

    Subclasses must specify the `consume_example` function.

    Arguments:
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
        self,
        num_proc: None | int = None,
        tqdm_kwargs: dict[str, Any] = {},
        tqdm_update_interval: float = 0.02,
    ):
        self.num_proc = num_proc or os.cpu_count()
        # tqdm arguments
        self.tqdm_kwargs = tqdm_kwargs
        self.tqdm_update_interval = tqdm_update_interval

    def consume(
        self, data: datasets.Dataset | datasets.IterableDataset
    ) -> Any:
        """Consume a given dataset

        Arguments:
            data (datasets.Dataset, datasets.IterableDataset):
                the dataset to consume
        """

        # convert dataset to iterable dataset
        if isinstance(data, datasets.Dataset):
            data = data.to_iterable_dataset(num_shards=self.num_proc)

        # create the shard queue and fill it with all shard ids
        shard_queue = mp.Queue()
        list(map(shard_queue.put, range(data.n_shards)))

        # spawn at most as many processes as there is work to
        # distribute among them
        num_proc = min(self.num_proc, data.n_shards)

        # create tqdm connections
        tqdm_conns = [mp.Pipe(duplex=False) for _ in range(num_proc)]
        # create tqdm worker
        tqdm_worker = mp.Process(
            name="%s[tqdm]" % type(self).__name__,
            target=self._tqdm_worker_fn,
            kwargs=dict(
                readers=[r for r, _ in tqdm_conns],
                **(self.tqdm_kwargs | {"total": data.n_shards, "unit": "sh"}),
            ),
            daemon=True,
        )
        tqdm_worker.start()

        # spawn all consumer workers and start them
        workers = [
            mp.Process(
                name="%s[worker_id=%i]" % (type(self).__name__, i),
                target=self._worker_fn,
                kwargs=dict(
                    worker_id=i,
                    shard_queue=shard_queue,
                    data=data,
                    tqdm_writer=tqdm_conns[i][1],
                ),
                daemon=True,
            )
            for i in range(num_proc)
        ]
        for w in workers:
            w.start()

        # wait for all consumer workers to finish
        for w in workers:
            w.join()
        # join the tqdm worker
        tqdm_worker.join()

    def _tqdm_worker_fn(
        self, readers: list[mp.connection.Connection], **kwargs
    ) -> None:
        """tqdm worker function

        Manages the tqdm progress bar for the consumer.

        Arguments:
            readers (list[mp.connection.Connection]):
                connections to consumer workers to listen to
            kwargs (Any): keyword arguments passed to tqdm
        """
        # create a progress bar
        pbar = tqdm(**kwargs)

        start_time = time()
        total_items_processed = 0

        while len(readers) > 0:
            # wait for any reader to receive data
            for r in mp.connection.wait(readers):
                data = r.recv()

                if data is None:
                    # corresponding worker terminated
                    r.close()
                    readers.remove(r)
                    continue

                # update data
                shard_finished, items_processed = data

                if shard_finished:
                    pbar.update()

                total_items_processed += items_processed
                throughput = total_items_processed / (time() - start_time)
                pbar.set_postfix_str("%.02fex/s" % throughput)

        # close the progress bar
        pbar.close()

    def _yield_shard_ids(self, shard_queue: mp.Queue) -> Iterable[int]:
        """Yield shard ids from a queue of shard ids to be processed"""
        while not shard_queue.empty():
            try:
                yield shard_queue.get(timeout=1)
            except Empty:
                pass

    def _worker_fn(
        self,
        worker_id: int,
        shard_queue: mp.Queue,
        data: datasets.IterableDataset,
        tqdm_writer: mp.connection.Connection,
    ) -> None:
        """Consumer worker function

        Implements general consumer loop and progress report to
        tqdm worker.

        The general consumer loop looks as follow:
            1. get dataset shard
            2. iterate over example in shard
            3. consume example

        Arguments:
            worker_id (int): worker id
            shard_queue (mp.Queue): queue of shard ids to process
            data (datasets.IterableDataset): dataset to consume shards of
            tqdm_writer (mp.connection.Connection): connection to tqdm worker
        """

        # initialize worker
        worker = mp.current_process()
        self.initialize_worker(worker, worker_id, data)
        # prepare dataset
        data = data._prepare_ex_iterable_for_iteration()

        try:
            # process each shard
            for shard_id in self._yield_shard_ids(shard_queue):
                shard = data.shard_data_sources(shard_id, data.n_shards)

                last_update_id = -1
                last_update_time = time()

                for example_id, (_, example) in enumerate(shard):
                    self.consume_example(
                        worker=worker,
                        worker_id=worker_id,
                        shard_id=shard_id,
                        example_id=example_id,
                        example=example,
                    )

                    if time() - last_update_time > self.tqdm_update_interval:
                        tqdm_writer.send((False, example_id - last_update_id))
                        last_update_id = example_id
                        last_update_time = time()

                # send final update for current shard
                tqdm_writer.send((True, example_id - last_update_id))

        finally:
            # finalize worker and tell tqdm worker to close the connection
            self.finalize_worker(worker, worker_id, data)
            tqdm_writer.send(None)

    def initialize_worker(
        self,
        worker: mp.Process,
        worker_id: int,
        data: datasets.IterableDataset,
    ) -> None:
        """Initialize worker

        Overwrite this function to implement logic to be executed once
        before starting the worker loop.

        Arguments:
            worker (mp.Process): worker process
            worker_id (int): worker id
        """
        pass

    def finalize_worker(
        self,
        worker: mp.Process,
        worker_id: int,
        data: datasets.IterableDataset,
    ) -> None:
        """Finalize worker

        Overwrite this function to implement logic to be executed once
        after the worker loop finished.

        Arguments:
            worker (mp.Process): worker process
            worker_id (int): worker id
        """
        pass

    @abstractmethod
    def consume_example(
        self,
        worker: mp.Process,
        worker_id: int,
        shard_id: int,
        example_id: int,
        example: dict[str, Any],
    ) -> None:
        """Abstract function to consume a given example

        This function implements the actual consume logic in subclasses.

        Arguments:
            worker (mp.Process): worker process
            worker_id (int): worker id
            shard_id (int): dataset shard id
            example_id (int): example id in the current dataset shard
            example (dict[str, Any]): the example to consume
        """
        ...
