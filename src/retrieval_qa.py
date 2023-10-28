"""
RetrievalQA
"""
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore

from src import CFG

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


def build_retrieval_qa(llm: LLM, vectordb: VectorStore) -> RetrievalQA:
    """Builds a retrieval QA model.

    Args:
        llm (CTransformers): The language model to use.
        vectordb (FAISS): The vector database to use.

    Returns:
        RetrievalQA: The retrieval QA model.
    """
    prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": CFG.SEARCH_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return retrieval_qa


def build_retrieval_chain(
    llm: LLM, vectordb: VectorStore
) -> ConversationalRetrievalChain:
    """Builds a conversational retrieval chain model.

    Args:
        llm (CTransformers): The language model to use.
        vectordb (FAISS): The vector database to use.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain model.
    """
    prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": CFG.SEARCH_K}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return retrieval_chain
