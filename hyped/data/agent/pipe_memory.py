from langchain.schema import BaseMemory
from langchain_core.messages import SystemMessage
from hyped.data.pipe import DataPipe
from hyped.data.agent.format import DataPipeFormatter, FeaturesFormatter
from datasets import Features
from typing import Any


class DataPipeMemory(BaseMemory):
    data_pipe: DataPipe
    in_features: Features
    memory_key: str = "data_pipe"

    def clear(self) -> None:
        pass

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        # build initial memory messages containing the description
        # of the current data pipeline and the initial dataset features
        messages = [
            SystemMessage(content=DataPipeFormatter.build_doc()),
            SystemMessage(
                content=DataPipeFormatter.build_desc(self.data_pipe)
            ),
            SystemMessage(
                content=(
                    "The dataset initially contains the following "
                    "features: %s"
                    % FeaturesFormatter.build_desc(self.in_features)
                )
            ),
        ]

        if len(self.data_pipe) == 0:
            # we dont need to add the current output
            # features when the data pipeline is empty
            return {self.memory_key: messages}

        try:
            # try to prepare the data pipeline to compute the output features
            out_features = self.data_pipe.prepare(self.in_features)
        except Exception as e:
            # this shouldn't fail as the tools fallback to the
            # last working version of the data pipeline when the
            # preparation fails, however when it does occur we
            # add the error message
            messages.append(
                SystemMessage(
                    content=(
                        "The current data pipeline throws the following "
                        "error: %s" % str(e)
                    )
                )
            )
            # return messages
            return {self.memory_key: messages}

        # add output features to messages
        messages.append(
            SystemMessage(
                content=(
                    "The current data pipeline maps these input features to "
                    "the following features: %s"
                    % FeaturesFormatter.build_desc(out_features)
                )
            )
        )

        # return messages
        return {self.memory_key: messages}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        pass
