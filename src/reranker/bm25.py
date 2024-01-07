from typing import Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from langchain.schema import Document
from langchain.pydantic_v1 import Extra

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


class BM25Reranker(BaseDocumentCompressor):
    """Reranker based on BM25."""

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("gpt2")
    top_n: int = 4

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
        Compress documents using gpt2 and BM25.

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
        # tokenize content for bm25 instance
        tokenized_content = self.tokenizer(docs).input_ids

        # tokenize query
        tokenized_query = self.tokenizer([query]).input_ids[0]

        bm25 = BM25Okapi(tokenized_content)
        scores = bm25.get_scores(tokenized_query)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]
