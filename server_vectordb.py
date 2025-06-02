import streamlit as st
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP

from src import CFG
from src.retrievers import create_rerank_retriever
from src.tools import create_retriever_tool, think
from src.vectordbs import load_faiss, load_chroma
from streamlit_app.utils import cache_base_embeddings, cache_reranker

EMBEDDING_FUNCTION = cache_base_embeddings()
RERANKER = cache_reranker()


@st.cache_resource
def cache_vectordb(vectordb_config: dict):
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(EMBEDDING_FUNCTION, vectordb_config["PATH"])
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(EMBEDDING_FUNCTION, vectordb_config["PATH"])
    raise NotImplementedError


tools = [to_fastmcp(think)]
for vectordb in CFG.VECTORDB:
    db = cache_vectordb(dict(vectordb))
    retriever = create_rerank_retriever(db, RERANKER)
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name=vectordb["NAME"],
        description=vectordb["DESCRIPTION"],
    )
    tools.append(to_fastmcp(retriever_tool))


mcp = FastMCP("VectorDB", tools=tools)


if __name__ == "__main__":
    mcp.run(transport="stdio")
