from typing import Any

from datasets import Dataset
from langchain.schema import BaseMemory
from langchain_core.messages import SystemMessage

from hyped.data.agent.format import DataPipeFormatter
from hyped.data.pipe import DataPipe


class DataPipeMemory(BaseMemory):
    data_pipe: DataPipe
    sample_ds: Dataset
    memory_key: str = "data_pipe"

    @property
    def data_pipe_formatter(self) -> DataPipeFormatter:
        return DataPipeFormatter(self.sample_ds)

    def clear(self) -> None:
        pass

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        # build initial memory messages containing the description
        # of the current data pipeline and the initial dataset features
        messages = [
            SystemMessage(
                content=(
                    "The initial dataset contains the following features: "
                    "%s\n Here is a list of samples of the dataset in json "
                    "format %s"
                    % (
                        str(self.sample_ds.features),
                        self.data_pipe_formatter.format_samples(
                            samples=self.sample_ds.to_dict()
                        ),
                    )
                )
                if len(self.sample_ds) > 0
                else (
                    "The initial dataset contains the following features: "
                    "%s" % str(self.sample_ds.features)
                )
            ),
            SystemMessage(
                content=(
                    "The data pipeline is a sequence of data processors. "
                    "A data processor implements modular operation on the "
                    "dataset, typically computing new features. By stacking "
                    "a set of data processors in a data pipeline, the input "
                    "features can be mapped to the expected output."
                )
            ),
            SystemMessage(
                content="%s\nData Pipeline: %s"
                % (
                    self.data_pipe_formatter.build_doc(),
                    self.data_pipe_formatter.build_desc(self.data_pipe),
                )
            ),
        ]

        # return messages
        return {self.memory_key: messages}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        pass
