import json

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException

from hyped.data.agent.pipe_tools.base import BaseDataPipeManipulationTool
from hyped.data.pipe import DataPipe


class UpdateDataProcessorInPipeInputs(BaseModel):
    index: int = Field(
        description=(
            "The index of the data processor to update in "
            "the current data pipeline"
        )
    )
    config: str = Field(
        description=(
            "A json string specifying new configuration of the "
            "data processor. For more information please refer to "
            "the configuration of the data processor."
        )
    )


class UpdateDataProcessorInPipe(BaseDataPipeManipulationTool):
    name: str = "UpdateDataProcessorInPipe"
    description: str = (
        "Update the configuration of a data processor in the current data pipe"
    )
    args_schema: type[BaseModel] = UpdateDataProcessorInPipeInputs

    def manipulate_data_pipe(
        self, data_pipe: DataPipe, index: int, config: str
    ) -> None:
        if index >= len(data_pipe):
            raise ToolException(
                "IndexOutOfBoundsError : the index %i does not refer to "
                "a data processor in the current data pipe of size %i"
                % (index, len(data_pipe))
            )

        # get current data processor and configuration
        cur_processor = data_pipe[index]
        cur_config = cur_processor.config.to_dict()
        # get the processor and configuration types
        processor_t = type(cur_processor)
        config_t = processor_t.config_type

        # create configuration
        try:
            # update current config with values from new config
            config = json.loads(config)
            cur_config.update(config)
            # build config object from dictionary
            config = config_t.from_dict(cur_config)
        except json.JSONDecodeError as e:
            raise ToolException("JsonDecodeError: %s" % e.args[0])
        except TypeError as e:
            raise ToolException(
                "%s. Please make sure to follow the documentation of the "
                "data processor." % e.args[0]
            )

        # create the new data pipe and overwrite the old one
        processor = processor_t.from_config(config)
        data_pipe[index] = processor
