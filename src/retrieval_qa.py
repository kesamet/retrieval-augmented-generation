"""
RetrievalQA
"""
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

from src import CFG
from src.llm import build_llm

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def build_retrieval_qa(llm, vectordb):
    prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"],
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": CFG.SEARCH_K}),
        return_source_documents=CFG.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": prompt},
    )
    return retrieval_qa
