"""
Embeddings
"""
from langchain.embeddings import HuggingFaceEmbeddings

from src import CFG


def build_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=CFG.EMBEDDINGS_MODEL,
        model_kwargs={"device": CFG.DEVICE},
    )
    return embeddings
