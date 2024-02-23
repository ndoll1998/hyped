from abc import ABC, abstractmethod
from copy import deepcopy

from datasets import Dataset
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool, ToolException

from hyped.data.agent.format import DataPipeFormatter
from hyped.data.pipe import DataPipe


class BaseDataPipeManipulationTool(BaseTool, ABC):
    # tool documentation
    name: str
    description: str
    args_schema: type[BaseModel]
    # data pipeline and input features
    data_pipe: DataPipe
    sample_ds: Dataset

    class Config:
        arbitrary_types_allowed: bool = True

    @property
    def data_pipe_formatter(self) -> DataPipeFormatter:
        return DataPipeFormatter(self.sample_ds)

    def _build_error_response(self, error: ToolException) -> str:
        return (
            "The following error occurred during tool execution: %s"
            % error.args[0]
        )

    def _build_response_description(self) -> str:
        return (
            "When successful, returns the current data pipeline "
            "in the following format: %s\n"
            "On error returns a description of the error and a hint "
            "on how to solve it. In this case the changes to the data "
            "pipeline are reverted." % self.data_pipe_formatter.build_doc()
        )

    def __init__(self, **kwargs) -> None:
        super(BaseDataPipeManipulationTool, self).__init__(**kwargs)
        # add response description to the tool description
        self.description = "%s\n\n%s" % (
            self.description,
            self._build_response_description(),
        )

    @abstractmethod
    def manipulate_data_pipe(
        self, data_pipe: DataPipe, *args, **kwargs
    ) -> None:
        pass

    def _run(
        self,
        *args,
        run_manager: None | CallbackManagerForToolRun = None,
        **kwargs,
    ) -> str:
        # create a copy of the data pipe which to manipulate
        data_pipe_copy = deepcopy(self.data_pipe)

        try:
            # manipulate data pipe
            self.manipulate_data_pipe(self.data_pipe, *args, **kwargs)

            try:
                # try to apply the current data pipeline to the
                # sample dataset and build the output response
                return self.data_pipe_formatter.build_desc(self.data_pipe)

            except KeyError as e:
                raise ToolException(
                    "%s. Please make sure that all features required for the "
                    "execution of the data processor are present." % e.args[0]
                )
            except TypeError as e:
                raise ToolException(
                    "%s. Please make sure that the input features to the data "
                    "processor are of the expected feature type." % e.args[0]
                )

        except ToolException as e:
            # reset data pipe and build error response
            self.data_pipe[:] = data_pipe_copy
            return self._build_error_response(e)

    async def _arun(
        self,
        *args,
        run_manager: None | AsyncCallbackManagerForToolRun = None,
        **kwargs,
    ) -> str:
        # TODO
        raise NotImplementedError()
