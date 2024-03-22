"""
Tracing with arize-phoenix[evals] (https://github.com/Arize-ai/phoenix)
"""

import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

from src.embeddings import build_base_embeddings
from src.vectordb import load_chroma
from src.reranker import build_reranker
from src.retrieval_qa import build_rerank_retriever
from src.llms import build_llm
from src.prompt_templates import QA_TEMPLATE

# Launch phoenix
session = px.launch_app()

# By default, the traces will be exported to the locally running Phoenix server.
tracer = LangChainInstrumentor().instrument()

# Setup RAG
embedding_function = build_base_embeddings()
vectordb = load_chroma(embedding_function)
reranker = build_reranker()
retriever = build_rerank_retriever(vectordb, reranker)
llm = build_llm()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # stuff, refine, map_reduce, and map_rerank
    retriever=retriever,
    # return_source_documents=True,
    chain_type_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
)

# Test run
query = "Compare dpo with rlhf?"
response = chain.invoke(query, callbacks=[tracer])
