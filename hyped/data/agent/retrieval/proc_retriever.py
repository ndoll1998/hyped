import inspect

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever

from hyped.base.registry import RootedTypeRegistryView
from hyped.data.agent.format import DataProcessorTypeFormatter
from hyped.data.agent.retrieval.np_vec_store import NumpyVectorStore
from hyped.data.processors.base import BaseDataProcessor


class DataProcessorsRetrieverInputs(BaseModel):
    query: str = Field(description="Query to look up in retriever")


class DataProcessorsRetriever(BaseTool):
    name: str = "DataProcessorsRetriever"
    description: str = (
        "Query this retriever to get a set of Data Processors most relevant "
        "to your needs. The response contains a list of relevant data "
        "processors in form of dictionaries of the following format: %s\n"
        "Make sure to ask for the most relevant processors in form of a "
        "question. Example: I need to remove a specific feature from the "
        "dataset, what processor could i use?"
        % DataProcessorTypeFormatter.build_doc()
    )
    args_schema: type[BaseModel] = DataProcessorsRetrieverInputs

    # vector store acting as database to query into
    retriever: VectorStoreRetriever

    def __init__(
        self,
        type_registry: RootedTypeRegistryView,
        embedding: Embeddings,
        **kwargs,
    ):
        # get all processor types
        assert issubclass(type_registry.root, BaseDataProcessor)
        processor_types = type_registry.concrete_types
        # make sure there are data processors registered
        if len(processor_types) == 0:
            raise RuntimeError("No data processors registered!")
        # build descriptive documents
        docs = list(map(self.build_processor_doc, processor_types))
        # create data processors vector store
        vector_store = NumpyVectorStore(docs, embedding)

        super(DataProcessorsRetriever, self).__init__(
            retriever=vector_store.as_retriever(**kwargs)
        )

    @property
    def processor_type_ids(self) -> list[str]:
        return [d.metadata["type_id"] for d in self.docs]

    def build_processor_doc(
        self, processor_type: type[BaseDataProcessor]
    ) -> str:
        return Document(
            # the document is embedded based on the page content
            page_content="{name}\n\n{docstr}".format(
                name=processor_type.__name__,
                docstr=inspect.getdoc(processor_type),
            ),
            # metadata contains the response given back to the model
            # when this document is retrieved
            metadata={
                "type": processor_type,
                "name": processor_type.__name__,
                "type_id": processor_type.config_type.t,
                "response": DataProcessorTypeFormatter.build_desc(
                    processor_type
                ),
            },
        )

    def _run(
        self,
        query: str,
        run_manager: None | CallbackManagerForToolRun = None,
    ) -> str:
        # retrieve the documents
        docs = self.retriever.get_relevant_documents(query)
        resp = ",\n".join([doc.metadata["response"] for doc in docs])
        # return json-formatted string of metadata
        return "[" + resp + "]"

    async def _arun(
        self,
        query: str,
        run_manager: None | AsyncCallbackManagerForToolRun = None,
    ) -> str:
        # retrieve the documents
        docs = await self.retriever.aget_relevant_documents(query)
        resp = ",\n".join([doc.metadata["response"] for doc in docs])
        # return json-formatted string of metadata
        return "[" + resp + "]"
