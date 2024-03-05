import json

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException

from hyped.data.agent.pipe_tools.base import BaseDataPipeManipulationTool
from hyped.data.pipe import DataPipe
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class AppendDataProcessorToPipeInputs(BaseModel):
    type_id: str = Field(description="The type_id of the processor to append")
    config: str = Field(
        description=(
            "A json string specifying the config of the data processor. "
            "For more information please refer to the config of the data "
            "processor to add."
        )
    )


class AppendDataProcessorToPipe(BaseDataPipeManipulationTool):
    name: str = "AppendDataProcessorToPipe"
    description: str = (
        "Append a new data processor to the current data pipeline."
    )
    args_schema: type[BaseModel] = AppendDataProcessorToPipeInputs

    def create_processor_instance(
        self, type_id: str, config: str
    ) -> BaseDataProcessor:
        if type_id not in BaseDataProcessorConfig.type_registry.type_ids:
            raise ToolException(
                "TypeIdNotFoundError: No processor with type id %s "
                "registered. Please ensure that your specified type "
                "id is valid."
            )

        # resolve type id
        config_t = BaseDataProcessorConfig.type_registry.get_type_by_t(type_id)
        processor_t = BaseDataProcessor.type_registry.get_type_by_t(
            "%s.impl" % type_id
        )

        # create configuration
        try:
            config = config_t.from_json(config)
        except json.JSONDecodeError as e:
            raise ToolException("JsonDecodeError: %s" % e.args[0])
        except TypeError as e:
            raise ToolException(
                "%s. Please make sure to follow the documentation of the "
                "data processor." % e.args[0]
            )

        # create data processor and add it to the pipe
        return processor_t.from_config(config)

    def manipulate_data_pipe(
        self, data_pipe: DataPipe, type_id: str, config: str
    ) -> None:
        # create processor and add it to the pipeline
        processor = self.create_processor_instance(type_id, config)
        data_pipe.append(processor)
