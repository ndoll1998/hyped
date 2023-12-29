import os
import multiprocessing as mp
from typing import Callable, Any


class SubprocessExecutor(object):
    """Subprocess Exector

    This class makes sure that a given function is always
    executed in a subprocess.

    If the executor is called from a subprocess, the function
    is executed directly. Otherwise a subprocess is created
    for the function execution.
    """

    def __init__(self):
        self.pid = os.getpid()
        self.queue = mp.Queue(maxsize=1)

    def execute(
        self,
        f: Callable[[Any, ...], Any],
        args: tuple[Any, ...],
        return_output: bool = True,
    ) -> Any:
        if os.getpid() != self.pid:
            return f(*args)

        elif not return_output:
            p = mp.Process(target=f, args=args)
            p.start()
            p.join()

        else:
            p = mp.Process(
                target=lambda q, args: q.put(f(*args)), args=(self.queue, args)
            )
            p.start()
            p.join()

            return self.queue.get()

    def __call__(self, *args, **kwargs) -> Any:
        return self.execute(*args, **kwargs)
