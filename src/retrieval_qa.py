"""
Retrieval QA
"""

from typing import List

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from src import CFG
from src.prompt_templates import QA_TEMPLATE, CONDENSE_QUESTION_TEMPLATE


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
                doc.metadata = {**doc.metadata, "similarity_score": score}
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            for doc, score in docs_and_similarities:
                doc.metadata = {**doc.metadata, "similarity_score": score}
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


def build_base_retriever(vectordb: VectorStore) -> VectorStoreRetriever:
    return VectorStoreRetrieverWithScores(
        vectorstore=vectordb, search_kwargs={"k": CFG.BASE_RETRIEVER_CONFIG.SEARCH_K}
    )


def build_multivector_retriever(
    vectorstore: VectorStore, docstore
) -> VectorStoreRetriever:
    from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_type=SearchType.mmr,
    )


def build_rerank_retriever(
    vectordb: VectorStore, reranker: BaseDocumentCompressor
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetrieverWithScores(
        vectorstore=vectordb, search_kwargs={"k": CFG.RERANK_RETRIEVER_CONFIG.SEARCH_K}
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )


def build_compression_retriever(
    vectordb: VectorStore, embeddings: Embeddings
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetrieverWithScores(
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


def build_retrieval_qa(llm: LLM, retriever: BaseRetriever) -> RetrievalQA:
    """Builds a retrieval QA model.

    Args:
        llm (LLM): The language model to use.
        retriever (BaseRetriever): The retriever to use.

    Returns:
        RetrievalQA: The retrieval QA model.
    """
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
    )
    return retrieval_qa


def build_retrieval_chain(
    vectordb: VectorStore, reranker: BaseDocumentCompressor, llm: LLM
) -> ConversationalRetrievalChain:
    """Builds a conversational retrieval chain model.

    Args:
        vectordb (VectorStore): The vector database to use.
        llm (LLM): The language model to use.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain model.
    """
    retriever = build_rerank_retriever(vectordb, reranker)

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
        condense_question_prompt=PromptTemplate.from_template(
            CONDENSE_QUESTION_TEMPLATE
        ),
    )
    return retrieval_chain
