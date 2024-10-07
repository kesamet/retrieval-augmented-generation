"""
Retrieval QA
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableBranch,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from src import CFG
from src.prompt_templates import QA_TEMPLATE, CONDENSE_QUESTION_TEMPLATE


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


def build_base_retriever(vectordb: VectorStore) -> VectorStoreRetriever:
    return VectorStoreRetriever(
        vectorstore=vectordb, search_kwargs={"k": CFG.BASE_RETRIEVER_CONFIG.SEARCH_K}
    )


def build_multivector_retriever(vectorstore: VectorStore, docstore) -> VectorStoreRetriever:
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
    base_retriever = VectorStoreRetriever(
        vectorstore=vectordb, search_kwargs={"k": CFG.RERANK_RETRIEVER_CONFIG.SEARCH_K}
    )
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


def build_compression_retriever(
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


def build_condense_question_chain(llm: LLM) -> Runnable:
    """Builds a chain that condenses question and chat history to create a standalone question."""
    condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    condense_question_chain = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input
            (lambda x: x["question"]),
        ),
        # If chat history, then we pass inputs to LLM chain
        condense_question_prompt | llm | StrOutputParser(),
    )
    return condense_question_chain


def build_question_answer_chain(llm: LLM) -> Runnable:
    """Builds a question-answer chain.
    Copied from langchain.chains.combine_documents.stuff.create_stuff_documents_chain
    """
    qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

    def format_docs(inputs: dict) -> str:
        return "\n\n".join(doc.page_content for doc in inputs["context"])

    question_answer_chain = (
        RunnablePassthrough.assign(context=format_docs).with_config(run_name="format_inputs")
        | qa_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")
    return question_answer_chain
