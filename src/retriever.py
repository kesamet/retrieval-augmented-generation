"""
Retriever
"""
from typing import Optional

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore

from src import CFG
from src.compressor import build_compression_retriever

if CFG.LLM_MODEL_TYPE == "llama":
    QA_TEMPLATE = """<s>[INST] <<SYS>> Use the following pieces of information to answer the user's question. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. <</SYS>>

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Answer:[/INST]"""
elif CFG.LLM_MODEL_TYPE == "mistral":
    QA_TEMPLATE = """<|im_start|>system
Use the following pieces of information to answer the user's question. \
If you don't know the answer, just say that you don't know, don't try to make up an answer.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}
Only return the helpful answer and nothing else.<|im_end|>
<|im_start|>assistant"""
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
    retriever = build_base_retriever(
        vectordb, use_compression=use_compression, embeddings=embeddings
    )

    prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )

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
    prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return retrieval_chain
