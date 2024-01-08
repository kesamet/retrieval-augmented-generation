"""
Retrieval QA
"""
from typing import Any

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever

from src import CFG

if CFG.PROMPT_TYPE == "llama":
    QA_TEMPLATE = """<s>[INST] <<SYS>> You are a helpful, respectful and honest AI Assistant. \
Use the following pieces of information to answer the user's question. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. <</SYS>>
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Answer: [/INST]"""
elif CFG.PROMPT_TYPE == "mistral":
    QA_TEMPLATE = """<s>[INST] You are a helpful, respectful and honest AI Assistant. \
Use the following pieces of information to answer the user's question. \
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer and nothing else.
Answer: [/INST]"""
else:
    raise NotImplementedError


def build_base_retriever(vectordb: VectorStore) -> VectorStoreRetriever:
    return vectordb.as_retriever(
        search_kwargs={"k": CFG.BASE_RETRIEVER_CONFIG.SEARCH_K}
    )


def build_rerank_retriever(vectordb: VectorStore) -> ContextualCompressionRetriever:
    from src.reranker import BGEReranker, TARTReranker

    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": CFG.RERANK_RETRIEVER_CONFIG.SEARCH_K}
    )
    if CFG.RERANKER_NAME == "BGE":
        reranker = BGEReranker(top_n=CFG.RERANK_RETRIEVER_CONFIG.TOP_N)
    elif CFG.RERANKER_NAME == "TART":
        reranker = TARTReranker(top_n=CFG.RERANK_RETRIEVER_CONFIG.TOP_N)
    else:
        raise NotImplementedError
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )


def build_compression_retriever(
    vectordb: VectorStore, embeddings: Embeddings
) -> ContextualCompressionRetriever:
    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": CFG.COMPRESSION_RETRIEVER_CONFIG.SEARCH_K}
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


def build_retrieval_qa(llm: LLM, retriever: Any) -> RetrievalQA:
    """Builds a retrieval QA model.

    Args:
        llm (LLM): The language model to use.
        retriever (Any): The retriever to use.

    Returns:
        RetrievalQA: The retrieval QA model.
    """
    prompt = PromptTemplate.from_template(QA_TEMPLATE)

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return retrieval_qa


def build_retrieval_chain(
    vectordb: VectorStore, llm: LLM
) -> ConversationalRetrievalChain:
    """Builds a conversational retrieval chain model.

    Args:
        vectordb (VectorStore): The vector database to use.
        llm (LLM): The language model to use.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain model.
    """
    retriever = build_rerank_retriever(vectordb)
    prompt = PromptTemplate.from_template(QA_TEMPLATE)

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return retrieval_chain
