import json
from typing import Any

from langchain.schema import BaseMemory
from langchain_core.messages import SystemMessage

from hyped.base.registry import RootedTypeRegistryView
from hyped.data.processors.base import BaseDataProcessor


class AvailableDataProcessorsMemory(BaseMemory):
    type_registry: RootedTypeRegistryView
    memory_key: str = "available_data_processors"

    def __init__(self, *args, **kwargs):
        super(AvailableDataProcessorsMemory, self).__init__(*args, **kwargs)
        assert issubclass(self.type_registry.root, BaseDataProcessor)

    @property
    def type_ids(self) -> list[str]:
        return [p.config_type.t for p in self.type_registry.concrete_types]

    def clear(self) -> None:
        pass

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return {
            self.memory_key: [
                SystemMessage(
                    content=(
                        "The following data processors are available: %s"
                        % json.dumps(self.type_ids)
                    )
                )
            ]
        }

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        pass
