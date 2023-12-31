from hyped.data.processors.statistics.report import (
    StatisticsReport,
    StatisticsReportStorage,
    statistics_report_manager,
)
from hyped.utils.executor import SubprocessExecutor
import multiprocessing as mp
import warnings
import pytest


def _is_lock_acquired(lock):
    # try to acquire lock
    could_acquire = lock.acquire(blocking=False)
    # release when axquired
    if could_acquire:
        lock.release()
    # lock was already acquired when it couldn't be acquired here
    return not could_acquire


def is_lock_acquired(lock):
    # requires to be checked in different process
    # because RLock objects can be acquired multiple
    # times by the same process
    with SubprocessExecutor() as executor:
        return executor.execute(f=_is_lock_acquired, args=(lock,))


class TestStatisticsReportStorage(object):
    @pytest.fixture
    def storage(self):
        with mp.Manager() as manager:
            yield StatisticsReportStorage(manager)

    def test_register_keys(self, storage) -> None:
        keys = "ABCDEF"
        # register keys and make sure all keys up to that
        # point are registered in the storage
        for i, k in enumerate(keys, 1):
            storage.register(k, 0)
            assert storage.registered_keys == set(keys[:i])

    def test_get_set_value(self, storage):
        # register keys
        for k in "ABCDEF":
            storage.register(k, k)
        # check values
        for k in storage.registered_keys:
            assert storage.get(k) == k

        for i in range(5):
            # update values for each statistic
            for k in storage.registered_keys:
                storage.set(k, k * i)
            # test updated values
            for k in storage.registered_keys:
                assert storage.get(k) == k * i

    def test_locks(self, storage):
        keys = "ABCDEF"
        # add some keys to the storage
        for i, k in enumerate(keys, 1):
            storage.register(k, 0)

        for k in keys:
            # acquire lock for current key
            with storage.get_lock_for(k):
                # check all locks
                for kk in keys:
                    # get state of lock for current test key
                    lock = storage.get_lock_for(kk)
                    # check if lock is locked in subprocess
                    is_locked = is_lock_acquired(lock)
                    # check expectation
                    if k == kk:
                        assert is_locked
                    else:
                        assert not is_locked


class TestStatisticsReportManager(object):
    @pytest.fixture
    def manager(self):
        return statistics_report_manager

    def test_activate_reports(self, manager):
        storages = [manager.new_statistics_report_storage() for _ in range(5)]

        # ignore warning that no statistics report is active
        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            for i, storage in enumerate(storages, 1):
                # activate report
                manager.activate(storage)
                # make sure all storages up to this point are active
                assert all(map(manager.is_active, storages[:i]))
                assert not any(map(manager.is_active, storages[i:]))
                # check reports iterator too
                assert all(s in storages[:i] for s in manager.reports)
                assert not any(s in storages[i:] for s in manager.reports)

            for i, storage in enumerate(storages, 1):
                # deactivate report
                manager.deactivate(storage)
                # make sure all storages up to this point ad unactive
                assert all(map(manager.is_active, storages[i:]))
                assert not any(map(manager.is_active, storages[:i]))
                # check reports iterator too
                assert all(s in storages[i:] for s in manager.reports)
                assert not any(s in storages[:i] for s in manager.reports)


class TestStatisticsReport(object):
    @pytest.fixture
    def manager(self):
        return statistics_report_manager

    @pytest.fixture
    def report(self):
        return StatisticsReport()

    def test_activate_report(self, manager, report):
        assert not manager.is_active(report.storage)

        # activate and deactivate report and make sure
        # it is tracked by the manager
        report.activate()
        assert manager.is_active(report.storage)
        report.deactivate()
        assert not manager.is_active(report.storage)

        # same using context manager
        with report:
            assert manager.is_active(report.storage)
        assert not manager.is_active(report.storage)

    def test_register_keys(self, report) -> None:
        keys = "ABCDEF"
        # register all keys to statistic storage underlying
        # the report object
        for i, k in enumerate(keys, 1):
            report.storage.register(k, k)
            # ensure keys are registered to report
            assert report.registered_keys == set(keys[:i])

        # check initial values
        for k in report.registered_keys:
            assert report.get(k) == k

        for i in range(5):
            # update values for each statistic
            for k in report.registered_keys:
                report.storage.set(k, k * i)
            # test updated values
            for k in report.registered_keys:
                assert report.get(k) == k * i
