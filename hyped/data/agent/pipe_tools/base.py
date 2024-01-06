from abc import ABC, abstractmethod
from copy import deepcopy
from datasets import Features
from hyped.data.pipe import DataPipe
from hyped.data.agent.format import DataPipeFormatter, FeaturesFormatter
from langchain_core.tools import BaseTool, ToolException
from langchain.pydantic_v1 import BaseModel, validator
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class BaseDataPipeManipulationTool(BaseTool, ABC):
    # tool documentation
    name: str
    description: str
    args_schema: type[BaseModel]
    # data pipeline and input features
    data_pipe: DataPipe
    in_features: Features

    def _build_response(self, output_features: Features) -> str:
        # build response
        return "Data Pipe: %s\nOutput Features: %s" % (
            DataPipeFormatter.build_desc(self.data_pipe),
            FeaturesFormatter.build_desc(output_features),
        )

    def _build_error_response(self, error: ToolException) -> str:
        return (
            "The following error occurred during tool execution: %s"
            % error.args[0]
        )

    @classmethod
    def _build_response_description(cls) -> str:
        return (
            "When successful, returns the current data pipeline "
            "and it's output features in the following format:\n"
            "Data Pipe: %s\nOutput Features: %s\n"
            "On error returns a description of the error and a hint "
            "on how to solve it. In this case the data pipeline is "
            "not changed."
            % (DataPipeFormatter.build_doc(), FeaturesFormatter.build_doc())
        )

    @validator("description")
    def _add_response_description(cls, v: str) -> str:
        return "%s\n\n%s" % (v, cls._build_response_description())

    @abstractmethod
    def manipulate_data_pipe(
        self, data_pipe: DataPipe, *args, **kwargs
    ) -> None:
        pass

    def _run(
        self,
        *args,
        run_manager: None | CallbackManagerForToolRun = None,
        **kwargs
    ) -> str:
        # create a copy of the data pipe which to manipulate
        data_pipe_copy = deepcopy(self.data_pipe)

        try:
            # manipulate data pipe
            self.manipulate_data_pipe(self.data_pipe, *args, **kwargs)

            try:
                # try to prepare pipeline to check features
                output_features = self.data_pipe.prepare(self.in_features)
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

            return self._build_response(output_features)

        except ToolException as e:
            # reset data pipe and build error response
            self.data_pipe[:] = data_pipe_copy
            return self._build_error_response(e)

    async def _arun(
        self,
        *args,
        run_manager: None | AsyncCallbackManagerForToolRun = None,
        **kwargs
    ) -> str:
        # TODO
        raise NotImplementedError()
