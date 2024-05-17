"""
Retrieval QA
"""

from typing import List

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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch
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
                doc.metadata = {**doc.metadata, "similarity_score": f"{score}:.4f"}
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            for doc, score in docs_and_similarities:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score:.4f}"}
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


def build_question_answer_chain(llm: LLM) -> Runnable:
    """Builds a question-answer chain.
    Copied from langchain.chains.combine_documents.stuff.create_stuff_documents_chain

    """
    qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

    def format_docs(inputs: dict) -> str:
        return "\n\n".join(doc.page_content for doc in inputs["context"])

    question_answer_chain = (
        RunnablePassthrough.assign(context=format_docs).with_config(
            run_name="format_inputs"
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")
    return question_answer_chain


def build_rag_chain(llm: LLM, retriever: BaseRetriever) -> Runnable:
    """Builds a retrieval RAG chain.
    Adapted from langchain.chains.retrieval.create_retrieval_chain

    Args:
        llm (LLM): The language model to use.
        retriever (BaseRetriever): The retriever to use.

    Returns:
        RetrievalQA: The retrieval QA model.
    """
    # retrieval_docs = (lambda x: x["question"]) | retriever
    # question_answer_chain = build_question_answer_chain(llm)
    # rag_chain = (
    #     RunnablePassthrough.assign(
    #         context=retrieval_docs.with_config(run_name="retrieve_documents"),
    #     ).assign(answer=question_answer_chain)
    # ).with_config(run_name="retrieval_chain")

    # FIXME: RetrievalQA deprecated
    from langchain.chains.retrieval_qa.base import RetrievalQA

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
        verbose=True,
    )
    return rag_chain


def build_conv_rag_chain(
    vectordb: VectorStore, reranker: BaseDocumentCompressor, llm: LLM
) -> Runnable:
    """Builds a conversational RAG chain.
    Adapted from langchain.chains.retrieval.create_retrieval_chain,
    langchain.chains.history_aware_retriever.create_history_aware_retriever

    Args:
        vectordb (VectorStore): The vector database to use.
        llm (LLM): The language model to use.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain model.
    """
    retriever = build_rerank_retriever(vectordb, reranker)

    # # From langchain.chains.history_aware_retriever.create_history_aware_retriever
    # condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    # history_aware_retriever = RunnableBranch(
    #     (
    #         # Both empty string and empty list evaluate to False
    #         lambda x: not x.get("chat_history", False),
    #         # If no chat history, then we just pass input to retriever
    #         (lambda x: x["question"]) | retriever,
    #     ),
    #     # If chat history, then we pass inputs to LLM chain, then to retriever
    #     condense_question_prompt | llm | StrOutputParser() | retriever,
    # )

    # question_answer_chain = build_question_answer_chain(llm)

    # # From langchain.chains.retrieval.create_retrieval_chain
    # rag_chain = (
    #     RunnablePassthrough.assign(
    #         context=history_aware_retriever.with_config(run_name="retrieve_documents"),
    #     ).assign(answer=question_answer_chain)
    # ).with_config(run_name="retrieval_chain")

    # FIXME: ConversationalRetrievalChain deprecated
    from langchain.chains.conversational_retrieval.base import (
        ConversationalRetrievalChain,
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
        condense_question_prompt=PromptTemplate.from_template(
            CONDENSE_QUESTION_TEMPLATE
        ),
        verbose=True,
    )
    return rag_chain


def condense_question_chain(llm: LLM):
    condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    chain = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input
            (lambda x: x["question"]),
        ),
        # If chat history, then we pass inputs to LLM chain
        condense_question_prompt | llm | StrOutputParser(),
    )
    return chain
