from typing import Any, Iterable, Optional

import numpy as np
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from numpy.typing import NDArray


class NumpyVectorStore(VectorStore):
    def __init__(
        self,
        docs: list[Document],
        embedding: Embeddings,
    ) -> None:
        self.docs = docs
        self.embedding = embedding

        self.embedding_matrix = self.embed(docs)

    def embed(self, docs: list[Document]) -> NDArray:
        texts = [doc.page_content for doc in docs]
        embeds = np.asarray(self.embedding.embed_documents(texts))
        embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
        return embeds

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[str]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
    ) -> VST:
        metadatas = (
            metadatas if metadatas is not None else ([None] * len(texts))
        )
        return cls(
            [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas)
            ],
            embedding,
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        k = min(k, len(self.docs))
        # embed query
        query_embed = np.asarray(self.embedding.embed_query(query))
        # embedding matrix is already normalized and we are
        # not interested in the actual scores but only the ranking,
        # thus the scaling by the norm of  the query embedding can
        # be ignored
        scores = (
            self.embedding_matrix @ query_embed
        )  # / np.linalg.norm(query_embed)
        # get the top-k documents
        idx = np.argpartition(scores, -k)[
            -k:
        ]  # unsorted top-k but runs in linear time
        idx = idx[np.argsort(-scores[idx])]  # sort top-k elements
        # return the documents to the top-k indices
        return [self.docs[i] for i in idx]
