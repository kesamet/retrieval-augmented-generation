"""
Tracing with arize-phoenix (https://github.com/Arize-ai/phoenix)
"""

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.embeddings import build_base_embeddings
from src.vectordbs import load_chroma
from src.rerankers import build_reranker
from src.retrievers import build_rerank_retriever
from src.llms import build_llm
from src.prompt_templates import QA_TEMPLATE

# Launch phoenix
session = px.launch_app()

tracer_provider = register(
    project_name="my-llm-app",  # Default is 'default'
    endpoint="http://localhost:6006/v1/traces",
)

# By default, the traces will be exported to the locally running Phoenix server.
tracer = LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Setup RAG
embedding_function = build_base_embeddings()
vectordb = load_chroma(embedding_function)
reranker = build_reranker()
retriever = build_rerank_retriever(vectordb, reranker)
llm = build_llm()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = PromptTemplate.from_template(QA_TEMPLATE)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough}
    | prompt
    | llm
    | StrOutputParser
)


# Test run
query = "Compare DPO with RLHF"
response = chain.invoke(query, callbacks=[tracer])
