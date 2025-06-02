import streamlit as st
from mcp.server.fastmcp import FastMCP
from langchain_core.tools.retriever import format_document
from langchain_core.prompts import PromptTemplate

from src import CFG
from src.retrievers import create_rerank_retriever
from src.vectordbs import load_faiss, load_chroma
from streamlit_app.utils import cache_base_embeddings, cache_reranker

mcp = FastMCP("VectorDB")

EMBEDDING_FUNCTION = cache_base_embeddings()
RERANKER = cache_reranker()

DOCUMENT_TEMPLATE = "{page_content}"


@st.cache_resource
def load_vectordb(vectordb_config: dict):
    if CFG.VECTORDB_TYPE == "faiss":
        return load_faiss(EMBEDDING_FUNCTION, vectordb_config["PATH"])
    if CFG.VECTORDB_TYPE == "chroma":
        return load_chroma(EMBEDDING_FUNCTION, vectordb_config["PATH"])
    raise NotImplementedError


@mcp.tool()
async def retrieve(query: str) -> str:
    """
    Retrieves information from the document database based on the query.

    Args:
        query (str): The search query to find relevant information

    Returns:
        str: Concatenated text content from all retrieved documents
    """
    vectordb = CFG.VECTORDB[0]
    db = load_vectordb(dict(vectordb))
    retriever = create_rerank_retriever(db, RERANKER)

    docs = retriever.invoke(query)
    document_prompt = PromptTemplate.from_template(DOCUMENT_TEMPLATE)
    return "\n\n".join(
        [format_document(doc, document_prompt) for doc in docs]
    )


@mcp.tool()
def think(thought: str):
    """
    Use the tool to think about something. It will not obtain new information,
    but just append the thought to the log.
    Use it when complex reasoning or some cache memory is needed.

    Args:
        thought (str): A thought to think about.
    """
    pass


if __name__ == "__main__":
    mcp.run(transport="stdio")
