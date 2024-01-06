from langchain.schema import BaseMemory
from langchain_core.messages import SystemMessage
from hyped.data.pipe import DataPipe
from hyped.data.agent.format import DataPipeFormatter
from typing import Any


class DataPipeMemory(BaseMemory):
    data_pipe: DataPipe
    memory_key: str = "data_pipe"

    def clear(self) -> None:
        pass

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return {
            self.memory_key: [
                SystemMessage(
                    content=DataPipeFormatter.build_desc(self.data_pipe)
                )
            ]
        }

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        pass
