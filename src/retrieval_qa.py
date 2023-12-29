"""
Retrieval QA
"""
from typing import Optional

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

from src import CFG

if CFG.PROMPT_TYPE == "llama":
    QA_TEMPLATE = """<s>[INST] <<SYS>>You are a helpful, respectful and honest AI Assistant. \
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


def build_base_retriever(
    vectordb: VectorStore,
    use_compression: bool = False,
    embeddings: Optional[Embeddings] = None,
):
    _retriever = vectordb.as_retriever(search_kwargs={"k": CFG.SEARCH_K})
    if use_compression:
        return build_compression_retriever(embeddings, _retriever)
    return _retriever


def build_compression_retriever(embeddings, retriever):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=CFG.SIMILARITY_THRESHOLD
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    return compression_retriever


def build_retrieval_qa(
    vectordb: VectorStore,
    llm: LLM,
    use_compression: bool = False,
    embeddings: Optional[Embeddings] = None,
) -> RetrievalQA:
    """Builds a retrieval QA model.

    Args:
        vectordb (VectorStore): The vector database to use.
        llm (LLM): The language model to use.
        use_compression (bool): To use contextual compression.
        embeddings (Embeddings): The embeddings model to use, if use_compression is True.

    Returns:
        RetrievalQA: The retrieval QA model.
    """
    # TODO: add reranker
    retriever = build_base_retriever(
        vectordb, use_compression=use_compression, embeddings=embeddings
    )

    prompt = PromptTemplate.from_template(template=QA_TEMPLATE)

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
    retriever = build_base_retriever(vectordb)
    prompt = PromptTemplate.from_template(template=QA_TEMPLATE)

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return retrieval_chain
