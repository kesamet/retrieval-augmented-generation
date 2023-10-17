"""
Build vectordb
"""
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from src import CFG
from src.embeddings import build_base_embeddings


def build_vectordb(filename: str) -> None:
    """Builds a vector database from a PDF file.

    Args:
        filename (str): Path to the PDF file.
    """
    documents = PyMuPDFLoader(filename).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.CHUNK_SIZE, chunk_overlap=CFG.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    embeddings = build_base_embeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(CFG.VECTORDB_FAISS_PATH)
