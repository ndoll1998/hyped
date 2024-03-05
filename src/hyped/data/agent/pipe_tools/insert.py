from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException

from hyped.data.agent.pipe_tools.append import AppendDataProcessorToPipe
from hyped.data.pipe import DataPipe


class InsertDataProcessorInPipeInputs(BaseModel):
    index: int = Field(
        description="The index where to insert the data processor"
    )
    type_id: str = Field(description="The type_id of the processor to insert")
    config: str = Field(
        description=(
            "A json string specifying the config of the data processor. "
            "For more information please refer to the config of the data "
            "processor to add."
        )
    )


class InsertDataProcessorInPipe(AppendDataProcessorToPipe):
    name: str = "InsertDataProcessorInPipe"
    description: str = (
        "Insert a new data processor at a specific index "
        "into the current data pipeline."
    )
    args_schema: type[BaseModel] = InsertDataProcessorInPipeInputs

    def manipulate_data_pipe(
        self, data_pipe: DataPipe, index: int, type_id: str, config: str
    ) -> None:
        if index < 0:
            raise ToolException("Negative index %i is not allowed" % index)

        if index > len(data_pipe):
            raise ToolException(
                "IndexOutOfBoundsError : the index %i exceeds the current "
                "data pipe of size %i" % (index, len(data_pipe))
            )

        # create data processor and insert it into the pipe
        processor = self.create_processor_instance(type_id, config)
        data_pipe.insert(index, processor)
