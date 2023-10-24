"""
Build vectordb
"""
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma

from src import CFG
from src.embeddings import build_base_embeddings


def build_vectordb(filename: str) -> None:
    """Builds a vector database from a PDF file.

    Args:
        filename (str): Path to the PDF file.
    """
    doc = PyMuPDFLoader(filename).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.CHUNK_SIZE, chunk_overlap=CFG.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(doc)

    embeddings = build_base_embeddings()
    if CFG.VECTORDB_TYPE == "faiss":
        _build_faiss(texts, embeddings)
    elif CFG.VECTORDB_TYPE == "chroma":
        _build_chroma(texts, embeddings)
    else:
        raise NotImplementedError


def _build_faiss(texts, embeddings):
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(CFG.VECTORDB_PATH)


def _build_chroma(texts, embeddings):
    _ = Chroma.from_documents(texts, embeddings, persist_directory=CFG.VECTORDB_PATH)


def load_faiss(embeddings):
    return FAISS.load_local(CFG.VECTORDB_PATH, embeddings)


def load_chroma(embeddings):
    return Chroma(persist_directory=CFG.VECTORDB_PATH, embedding_function=embeddings)
