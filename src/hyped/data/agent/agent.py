from datasets import Dataset
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import CombinedMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessage,
)

from hyped.data.agent.memories.pipe_memory import DataPipeMemory
from hyped.data.agent.memories.proc_memory import AvailableDataProcessorsMemory
from hyped.data.agent.pipe_toolkit import DataPipeManipulationToolkit
from hyped.data.agent.retrieval.proc_retriever import DataProcessorsRetriever
from hyped.data.pipe import DataPipe

# tools and memory
from hyped.data.processors.base import BaseDataProcessor


class OpenAIDataAgent(AgentExecutor):
    def __init__(
        self,
        llm: ChatOpenAI,
        emb: OpenAIEmbeddings,
        data_pipe: DataPipe,
        sample_ds: Dataset,
        **kwargs,
    ) -> None:
        # get the processors type registry containing
        # available processor types
        proc_type_registry = BaseDataProcessor.type_registry
        # create toolset
        tools = [
            DataProcessorsRetriever(
                proc_type_registry, emb, search_kwargs={"k": 3}
            ),
            *DataPipeManipulationToolkit(
                data_pipe=data_pipe,
                sample_ds=sample_ds,
            ).get_tools(),
        ]
        # create agent
        agent = create_openai_functions_agent(llm, tools, self.prompt)

        super(OpenAIDataAgent, self).__init__(
            agent=agent,
            tools=tools,
            memory=CombinedMemory(
                memories=[
                    AvailableDataProcessorsMemory(
                        type_registry=proc_type_registry
                    ),
                    DataPipeMemory(data_pipe=data_pipe, sample_ds=sample_ds),
                ]
            ),
            **kwargs,
        )

    @property
    def prompt(self) -> ChatPromptTemplate:
        return (
            SystemMessage(
                content=(
                    "You are a data scientist who is concerned with the "
                    "preprocessing of datasets. You will be given a set of "
                    "input features and, given the tools you are provided, "
                    "need to adapt the data pipeline to map these input "
                    "features to the user requirements. You will be given "
                    "a textual description of what the required output "
                    "features should be. "
                )
            )
            + MessagesPlaceholder(variable_name="available_data_processors")
            + MessagesPlaceholder(variable_name="data_pipe")
            + HumanMessagePromptTemplate.from_template(
                input_variables=["task"],
                template=(
                    "Please adapt the data pipeline to solve the following "
                    "task:\n{task}"
                ),
            )
            + MessagesPlaceholder(variable_name="agent_scratchpad")
        )
