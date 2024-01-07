from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessage,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# tools and memory
from hyped.data.agent.pipe_toolkit import DataPipeManipulationToolkit
from hyped.data.agent.retrieval.proc_retriever import DataProcessorsRetriever
from hyped.data.agent.pipe_memory import DataPipeMemory

from datasets import Features
from hyped.data.pipe import DataPipe


class OpenAIDataAgent(AgentExecutor):
    def __init__(
        self,
        llm: ChatOpenAI,
        emb: OpenAIEmbeddings,
        data_pipe: DataPipe,
        in_features: Features,
        **kwargs
    ) -> None:
        # create toolset
        tools = [
            DataProcessorsRetriever(emb, search_kwargs={"k": 3}),
            *DataPipeManipulationToolkit(
                data_pipe=data_pipe,
                in_features=in_features,
            ).get_tools(),
        ]
        # create agent
        agent = create_openai_functions_agent(llm, tools, self.prompt)

        super(OpenAIDataAgent, self).__init__(
            agent=agent,
            tools=tools,
            memory=DataPipeMemory(
                data_pipe=data_pipe, in_features=in_features
            ),
            **kwargs
        )

    @property
    def prompt(self) -> ChatPromptTemplate:
        return (
            SystemMessage(
                content=(
                    "You are a data scientist who is concerned with the "
                    "preprocessing of datasets. You will be given a set of "
                    "input features and, given the tools you are provided, "
                    "need to build a data pipeline to convert these input "
                    "features to meet the desired output. You will be given "
                    "a textual description of what the required output "
                    "features should be."
                    "The data pipeline is a sequence of data processors. "
                    "A data processor implements modular operation on the "
                    "dataset, typically computing new features. By stacking "
                    "a set of data processors in a data pipeline, the input "
                    "features can be mapped to the expected output."
                )
            )
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
