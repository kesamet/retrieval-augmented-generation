"""
Retrievers
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from src import CFG


# FIXME
class VectorStoreRetrieverWithScores(VectorStoreRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Adapted from https://github.com/langchain-ai/langchain/blob/2f8dd1a1619f25daa4737df4d378b1acd6ff83c4/
        libs/core/langchain_core/vectorstores.py#L692
        """
        if self.search_type == "similarity":
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
            for doc, score in docs_and_scores:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score}:.4f"}
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
            for doc, score in docs_and_similarities:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score:.4f}"}
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


def create_base_retriever(vectordb: VectorStore) -> VectorStoreRetriever:
    return VectorStoreRetriever(
        vectorstore=vectordb, search_kwargs={"k": CFG.BASE_RETRIEVER_CONFIG.SEARCH_K}
    )


def create_multivector_retriever(vectorstore: VectorStore, docstore) -> VectorStoreRetriever:
    from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_type=SearchType.mmr,
    )


def create_rerank_retriever(
    vectordb: VectorStore, reranker: BaseDocumentCompressor
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetriever(
        vectorstore=vectordb, search_kwargs={"k": CFG.RERANK_RETRIEVER_CONFIG.SEARCH_K}
    )
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


def create_compression_retriever(
    vectordb: VectorStore, embeddings: Embeddings
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetriever(
        vectorstore=vectordb,
        search_kwargs={"k": CFG.COMPRESSION_RETRIEVER_CONFIG.SEARCH_K},
    )

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=CFG.COMPRESSION_RETRIEVER_CONFIG.SIMILARITY_THRESHOLD,
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )
    return compression_retriever
