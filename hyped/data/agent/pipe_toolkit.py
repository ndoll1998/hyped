from datasets import Dataset
from langchain_community.agent_toolkits.base import BaseToolkit

from hyped.data.agent.pipe_tools.append import AppendDataProcessorToPipe
from hyped.data.agent.pipe_tools.base import BaseDataPipeManipulationTool
from hyped.data.agent.pipe_tools.insert import InsertDataProcessorInPipe
from hyped.data.agent.pipe_tools.remove import RemoveDataProcessorFromPipe
from hyped.data.agent.pipe_tools.update import UpdateDataProcessorInPipe
from hyped.data.pipe import DataPipe


class DataPipeManipulationToolkit(BaseToolkit):
    data_pipe: DataPipe = DataPipe()
    sample_ds: Dataset

    class Config:
        arbitrary_types_allowed: bool = True

    def get_tools(self) -> list[BaseDataPipeManipulationTool]:
        return [
            AppendDataProcessorToPipe(
                data_pipe=self.data_pipe, sample_ds=self.sample_ds
            ),
            InsertDataProcessorInPipe(
                data_pipe=self.data_pipe, sample_ds=self.sample_ds
            ),
            UpdateDataProcessorInPipe(
                data_pipe=self.data_pipe, sample_ds=self.sample_ds
            ),
            RemoveDataProcessorFromPipe(
                data_pipe=self.data_pipe, sample_ds=self.sample_ds
            ),
        ]
