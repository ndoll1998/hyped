from __future__ import annotations
import os
import warnings
import multiprocessing as mp
from copy import deepcopy
from typing import Any, Iterable
from hyped.utils.executor import SubprocessExecutor


class StatisticsReportStorage(object):
    """Statistics Report Storage

    Internal class implementing a thread-safe storage for statistics
    reports. It manages the values and locks for statistics.

    Arguments:
        manager (mp.Manager):
            multiprocessing manager responsible for sharing statistic
            values and locks between processes
    """

    def __init__(self, manager: mp.Manager):
        # safe main process id
        self.pid = os.getpid()
        # create shared storage for statistic values and locks
        self.manager = manager
        self.stats = self.manager.dict()
        self.locks = self.manager.dict()
        # create a subprocess executor used to interact
        # with the values and locks storages
        self.executor = SubprocessExecutor()
        # keep track of all registered keys
        self.registered_keys: set[str] = set()

    def __del__(self) -> None:
        # make sure to stop the executor when the storage is deleted
        if hasattr(self, "executor"):
            self.executor.stop()

    def _getter(self, d: dict[str, Any], k: str) -> Any:
        return self.executor(f=d.__getitem__, args=(k,))

    def _setter(self, d: dict[str, Any], k: str, v: Any) -> None:
        return self.executor(f=d.__setitem__, args=(k, v), return_output=False)

    def register(self, key: str, init_val: Any) -> None:
        """Register a statistic key to the storage

        Adds the initial value to the value storage and creates a lock
        dedicated to the statistic.

        Note that only the main process (i.e. the process that created the
        instance of the storage) can register new statistics to the storage.

        Arguments:
            key (str): statistic key under which to store the statistic
            init_val (Any): initial value of the statistic
        """
        # can only register keys from main process
        if os.getpid() != self.pid:
            raise RuntimeError(
                "Error while registering statistic key `%s`: "
                "cannot register key from subprocess" % key
            )
        # key already registered
        if key in self:
            raise RuntimeError(
                "Error while registering statistic key `%s`: "
                "Key already registered" % key
            )
        # create lock for the statistic
        lock = self.manager.RLock()
        # write initial value and lock to dicts
        self._setter(self.stats, key, deepcopy(init_val))
        self._setter(self.locks, key, lock)
        # add key to registered keys
        self.registered_keys.add(key)

    def get_lock_for(self, key: str) -> mp.RLock:
        """Get the lock dedicated to a given statistic

        Arguments:
            key (str): statistic key

        Returns.
            lock (mp.RLock): multiprocessing lock dedicated to the statistic
        """
        # key not registered
        if key not in self:
            raise KeyError(
                "Error while getting lock for statistic: "
                "Statistic key `%s` not registered" % key
            )
        # get lock for key
        return self._getter(self.locks, key)

    def get(self, key: str) -> Any:
        """Get the value of a given statistic

        Arguments:
            key (str): statistic key

        Returns:
            val (Any): statistic value
        """
        # key not registered
        if key not in self:
            raise KeyError(
                "Error while getting value for statistic: "
                "Statistic key `%s` not registered" % key
            )
        # get statistic value for key
        return self._getter(self.stats, key)

    def set(self, key: str, val: Any) -> None:
        """Set the value of a given statistic

        Arguments:
            key (str): statistic key
            val (Any): value to store in the statistic
        """
        # key not registered
        if key not in self:
            raise KeyError(
                "Error while setting value for statistic: "
                "Statistic key `%s` not registered" % key
            )
        # update value in statistics dict
        with self.get_lock_for(key):
            self._setter(self.stats, key, val)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, val: Any) -> None:
        self.set(key, val)

    def __contains__(self, key: str) -> bool:
        return key in self.registered_keys


class StatisticsReportManager(object):
    """Statistics Report Manager

    Internal class managing statistic report storages and the multiprocessing
    manager underlying the storages. It keeps track of the active reports,
    i.e. reports to which statistics should be written.
    """

    def __init__(self):
        self._manager: None | mp.Manager = None
        self._active_reports: set[StatisticsReportStorage] = set()

    @property
    def manager(self) -> mp.Manager:
        """Multiprocessing manager instance"""
        # TODO: when should the manager be shutdown
        # create new manager if needed
        if self._manager is None:
            self._manager = mp.Manager()
        # return the current manager
        return self._manager

    @property
    def reports(self) -> Iterable[StatisticsReportStorage]:
        """Iterator over active report storages

        Warns when no reports are activated

        Returns:
            reports_iter (Iterable[StatisticsReportStorage]):
                iterator over active report storages
        """

        # warn when no reports are active
        if len(self._active_reports) == 0:
            warnings.warn(
                "No active statistic reports found. Computed statistics will "
                "not be tracked. Active a `StatisticReport` instance to "
                "track statistics.",
                UserWarning,
            )
        # iterate over active reports
        return iter(self._active_reports)

    def new_statistics_report_storage(self) -> StatisticsReportStorage:
        """Create a new statistic report storage

        Returns:
            storage (StatisticReportStorage): new storage instance
        """
        return StatisticsReportStorage(self.manager)

    def is_active(self, report: StatisticsReportStorage) -> bool:
        """Check if a given statistic report storage is active

        Arguments:
            storage (StatisticsReportStorage): storage instance to check for

        Returns:
            is_active (bool):
                boolean indicating whether the storage is active or not
        """
        return report in self._active_reports

    def activate(self, report: StatisticsReportStorage) -> None:
        """Activate a given statistics report storage in order for it to track
        computed statistics.

        Arguments:
            storage (StatisticsReportStorage): storage to activate
        """
        self._active_reports.add(report)

    def deactivate(self, report: StatisticsReportStorage) -> None:
        """Deactivate a given statistics report storage

        Arguments:
            storage (StatisticsReportStorage): storage to deactivate
        """
        if self.is_active(report):
            self._active_reports.remove(report)


# create an instance of the statistics report manager
statistics_report_manager = StatisticsReportManager()


class StatisticsReport(object):
    """Statistics Report

    Tracks statistics computed in data statistics processors. Activate the
    report to start tracking statistics.

    Can be used as a context manager.
    """

    def __init__(self) -> None:
        self.storage = (
            statistics_report_manager.new_statistics_report_storage()
        )

    @property
    def registered_keys(self) -> set[str]:
        """Set of registered statistic keys"""
        return set(self.storage.registered_keys)

    def get(self, key: str) -> Any:
        """Get the statistic value to a given key

        Arguments:
            key (str): statistic key

        Returns:
            val (Any): value of the statistic
        """
        return self.storage.get(key)

    def activate(self) -> None:
        """Activate the report"""
        statistics_report_manager.activate(self.storage)

    def deactivate(self) -> None:
        """Deactivate the report"""
        statistics_report_manager.deactivate(self.storage)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.registered_keys

    def __enter__(self) -> StatisticsReport:
        self.activate()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.deactivate()

    def __str__(self) -> str:
        return "\n".join(
            ["%s: %s" % (k, self.get(k)) for k in self.registered_keys]
        )
