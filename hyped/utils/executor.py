from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

STOP_SIGNAL = "stop_signal"


class Status(IntEnum):
    NOT_STARTED = 0
    RUNNING = 1
    STOPPED = 2


@dataclass
class Task(object):
    f: Callable[[Any, ...], Any | None]
    args: tuple[Any]
    return_output: bool = False


class SubprocessExecutor(mp.Process):
    """Subprocess Exector

    This class makes sure that a given function is always
    executed in a subprocess.

    If the executor is called from a subprocess, the function
    is executed directly. Otherwise a subprocess is created
    for the function execution.
    """

    def __init__(self, *args, **kwargs):
        self.parent_pid = os.getpid()
        # save arguments for process initializer
        self.args = args
        self.kwargs = kwargs
        # flag to keep track if process is running
        self.status = Status.NOT_STARTED

    def init(self):
        # mark process as running
        self.status = Status.RUNNING
        # queues
        self.tasks = mp.Queue(maxsize=1)
        self.outs = mp.Queue(maxsize=1)
        self.errs = mp.Queue(maxsize=1)
        # events
        self.done = mp.Event()
        self.err = mp.Event()
        # initialize process
        super(SubprocessExecutor, self).__init__(*self.args, **self.kwargs)
        self.daemon = True
        # start process
        self.start()

    def run(self) -> None:
        """Worker function executed in process"""
        for task in iter(self.tasks.get, STOP_SIGNAL):
            assert not self.done.is_set()

            try:
                # try to execute task
                output = task.f(*task.args)
            except Exception as e:
                # catch error and put to error queue
                self.err.set()
                self.errs.put(e)
                self.done.set()
                continue

            # put return value to output queue
            if task.return_output:
                self.outs.put(output)

            # task finished
            self.done.set()

    def execute(
        self,
        f: Callable[[Any, ...], Any],
        args: tuple[Any, ...],
        return_output: bool = True,
    ) -> Any:
        if os.getpid() != self.parent_pid:
            return f(*args)

        # start process
        if self.status == Status.NOT_STARTED:
            self.init()

        # clear events
        self.done.clear()
        self.err.clear()
        # put task to queue
        self.tasks.put(Task(f, args, return_output))
        # wait for completion
        self.done.wait()

        # raise error from sub-process
        if self.err.is_set():
            raise self.errs.get()

        # return output when asked for
        if return_output:
            return self.outs.get()

    def stop(self) -> None:
        if self.status == Status.RUNNING:
            self.status = Status.STOPPED

            try:
                # try to send stop signal, might fail when
                # object is deleted at interpreter shutdown
                self.tasks.put(STOP_SIGNAL)
                self.join()
            except RuntimeError:
                pass

    def __del__(self) -> None:
        self.stop()

    def __call__(self, *args, **kwargs) -> Any:
        return self.execute(*args, **kwargs)

    def __enter__(self) -> SubprocessExecutor:
        self.init()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.stop()
