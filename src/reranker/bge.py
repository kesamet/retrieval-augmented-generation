from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

from langchain.schema import Document
from langchain.pydantic_v1 import Extra

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder

from src import CFG


class BGEReranker(BaseDocumentCompressor):
    """Reranker based on BGE-reranker (https://huggingface.co/BAAI/bge-reranker-base)."""

    model_path: str = os.path.join(CFG.MODELS_DIR, CFG.RERANKER_PATH)
    top_n: int = 4
    """Number of documents to return."""
    model: CrossEncoder = CrossEncoder(model_path)
    """CrossEncoder instance to use for reranking."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using BAAI/bge-reranker models.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results

    def rerank(self, query: str, docs: Sequence[str]) -> Sequence[Tuple[int, float]]:
        """
        Reranks a list of documents based on a given query using a pre-trained model.

        Args:
            query: The query string.
            docs: The list of documents to be reranked.

        Returns:
            A list of tuples containing the index of the document and its reranked score.
        """
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]
