from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException

from hyped.data.agent.pipe_tools.base import BaseDataPipeManipulationTool
from hyped.data.pipe import DataPipe


class RemoveDataProcessorFromPipeInputs(BaseModel):
    index: int = Field(
        description=(
            "The index of the data processor to remove from "
            "the current data pipeline"
        )
    )


class RemoveDataProcessorFromPipe(BaseDataPipeManipulationTool):
    name: str = "RemoveDataProcessorInPipe"
    description: str = "Remove a data processor from the current data pipe"
    args_schema: type[BaseModel] = RemoveDataProcessorFromPipeInputs

    def manipulate_data_pipe(self, data_pipe: DataPipe, index: int) -> None:
        if index >= len(data_pipe):
            raise ToolException(
                "IndexOutOfBoundsError : the index %i does not refer to "
                "a data processor in the current data pipe of size %i"
                % (index, len(data_pipe))
            )

        data_pipe.pop(index)
