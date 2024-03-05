import multiprocessing as mp
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from datasets import Features

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.data.processors.statistics.report import (
    StatisticsReportStorage,
    statistics_report_manager,
)


@dataclass
class BaseDataStatisticConfig(BaseDataProcessorConfig):
    """Base Statistic Data Processor Config

    Attributes:
        statistic_key (str):
            key under which the statistic is stored in reports.
            See `StatisticsReport` for more information.
    """

    t: Literal[
        "hyped.data.processors.statistics.base"
    ] = "hyped.data.processors.statistics.base"

    statistic_key: str = None

    def __post_init__(self) -> None:
        assert self.keep_input_features
        assert self.output_format is None


T = TypeVar("T", bound=BaseDataStatisticConfig)
U = TypeVar("U")


class BaseDataStatistic(BaseDataProcessor[T], Generic[T, U]):
    """Abstract Base Statistic Data Processor

    Provides basic functionality to implement data statistics
    as data processors. Sub-types need to define the abstract
    `initial_value`, `check_features` and `compute` functions.

    The computation of the statistic and the update to the
    statistics report object is implemented through the
    following pipeline:

    `extract` -> (`lock.acquire`) -> `compute` -> `update` -> (`lock.release`)

    Please refer to the documentation of the functions for
    further information.

    Arguments:
        config (BaseDataStatisticConfig): data statistic configuration
    """

    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Internal batch process

        Updates the statistic in all active reports given the batch of
        examples. The actual computations are outsourced to the
        `update_statistic` function.

        The return values of this function are constant, reflecting that
        statistic computations do not change the example values.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): empty dictionary
            src_index (list[int]): enumeration of input examples
        """

        extracted_value = self.extract(examples, index, rank)
        # update reports
        for report in statistics_report_manager.reports:
            # make sure the statistic is registered
            if self.config.statistic_key not in report:
                raise RuntimeError(
                    "Statistic not registered, make sure to prepare all "
                    "statistic processors after activating all reports"
                )
            # update statistic
            with report.get_lock_for(self.config.statistic_key):
                self.update(
                    report,
                    self.compute(
                        report[self.config.statistic_key],
                        extracted_value,
                    ),
                )

        # do nothing to the examples
        return {}, range(len(index))

    def map_features(self, features: Features) -> Features:
        """Map features function

        This function checks the input features and registers the
        statistic key to all active reports. The statistic key is
        specified in the configuration.

        As a statistic does not compute any features, the return
        value of this function is an empty feature dictionary.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): empty feature dictionary
        """
        # check input features
        self.check_features(features)
        # register statistic to all active reports
        for report in statistics_report_manager.reports:
            report.register(
                self.config.statistic_key,
                self.initial_value(features, report.manager),
            )

        return Features()

    @abstractmethod
    def check_features(self, features: Features) -> None:
        """Abstract check features function. Checks validity of input features

        Arguments:
            features (Features): input dataset features
        """
        ...

    @abstractmethod
    def initial_value(self, features: Features, manager: mp.Manager) -> U:
        """Abstract initial value function. Computes the initial value for
        the statistic.

        Arguments:
            features (Features): input dataset features
            manager (mp.Manager): multiprocessing manager

        Returns:
            init_val (Any): value used to initialize the statistic
        """
        ...

    def extract(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
    ) -> Any:
        """Extract relevant values from the batch of examples.

        This function is ment to handle all the preparation of the batch
        of examples. It returns all the extracted values required to compute
        the statistic. These values are directly piped to the ´compute´
        function.

        By default, the extract function returns the batch of examples.
        This way all logic is implemented in the `compute` function, which
        might lead to longer execution times due to blocking.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples
            index (list[int]): dataset indices of the batch of examples
            rank (int): execution process rank

        Returns:
            ext (Any): extracted values relevant for the update
        """
        return examples

    @abstractmethod
    def compute(
        self,
        val: U,
        ext: Any,
    ) -> U:
        """Abstract compute statistic function. Computes the updated statistic
        value based on the values extracted from the examples using the
        `extract` function.

        Arguments:
            val (Any): current statistic value
            ext (Any): the output of the `extract` function

        Returns:
            new_val (Any): new statistic value

        """
        ...

    def update(self, report: StatisticsReportStorage, val: U) -> None:
        """Update report storage with new value computed by `compute` function.

        Overwrite to implement more complex behaviors.

        Arguments:
            report (StatisticsReportStorage):
                report storage to update the statistic in
            val (Any): new statistic value computed by `compute` function
        """
        report.set(self.config.statistic_key, val)
