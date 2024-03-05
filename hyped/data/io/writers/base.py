import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from queue import Empty
from time import time
from typing import Any, Iterable

import datasets
from tqdm import tqdm
from tqdm.std import EMA


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
        tqdm_update_interval: float = 0.5,
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

        # render tqdm bar
        self._tqdm(readers=[r for r, _ in tqdm_conns], total=data.n_shards)

        # wait for all consumer workers to finish
        for w in workers:
            w.join()

    def _tqdm(
        self, readers: list[mp.connection.Connection], total: int
    ) -> None:
        """tqdm bar

        Manages the tqdm progress bar for the consumer.

        Arguments:
            readers (list[mp.connection.Connection]):
                connections to consumer workers to listen to
            kwargs (Any): keyword arguments passed to tqdm
        """
        # create a progress bar
        kwargs = self.tqdm_kwargs | {
            "total": total,
            "unit": "sh",
        }
        pbar = tqdm(**kwargs)

        dn_shards = 0
        dn_examples = 0
        total_examples = 0
        # exponential moving averages for items throughput
        ema_dn = EMA(smoothing=kwargs.get("smoothing", 0.3))
        ema_dt = EMA(smoothing=kwargs.get("smoothing", 0.3))

        while len(readers) > 0:
            # wait for any reader to receive data
            for r in mp.connection.wait(
                readers, timeout=self.tqdm_update_interval
            ):
                data = r.recv()

                if data is None:
                    # corresponding worker terminated
                    r.close()
                    readers.remove(r)
                    continue

                # update data
                shard_finished, examples_processed = data
                dn_shards += int(shard_finished)
                dn_examples += examples_processed
                total_examples += examples_processed

            t = pbar._time()
            dt = t - pbar.last_print_t

            if dt >= self.tqdm_update_interval:
                # compute examples throughput and update tqdm postfix
                throughput = ema_dn(dn_examples) / ema_dt(dt)
                formatted_total_examples = (
                    "%d" if total_examples < 10**6 else "%.2e"
                ) % total_examples
                pbar.set_postfix_str(
                    "%.02fex/s, %sex" % (throughput, formatted_total_examples),
                    refresh=False,
                )
                # update progress and refresh bar
                if pbar.update(dn_shards) is None:
                    pbar.refresh(lock_args=pbar.lock_args)
                    pbar.last_print_t = t
                # reset values
                dn_shards = 0
                dn_examples = 0

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
