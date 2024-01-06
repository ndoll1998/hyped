from datasets import Features
from hyped.data.pipe import DataPipe
from hyped.data.agent.pipe_tools.base import BaseDataPipeManipulationTool
from hyped.data.agent.pipe_tools.append import AppendDataProcessorToPipe
from hyped.data.agent.pipe_tools.update import UpdateDataProcessorInPipe
from hyped.data.agent.pipe_tools.remove import RemoveDataProcessorFromPipe
from langchain_community.agent_toolkits.base import BaseToolkit


class DataPipeManipulationToolkit(BaseToolkit):
    data_pipe: DataPipe = DataPipe()
    in_features: Features

    def get_tools(self) -> list[BaseDataPipeManipulationTool]:
        return [
            AppendDataProcessorToPipe(
                data_pipe=self.data_pipe, in_features=self.in_features
            ),
            UpdateDataProcessorInPipe(
                data_pipe=self.data_pipe, in_features=self.in_features
            ),
            RemoveDataProcessorFromPipe(
                data_pipe=self.data_pipe, in_features=self.in_features
            ),
        ]
