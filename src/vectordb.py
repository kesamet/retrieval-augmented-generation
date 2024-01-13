"""
VectorDB
"""
from typing import Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS, Chroma

from src import CFG
from src.embeddings import build_base_embeddings


def build_vectordb(filename: str) -> None:
    """Builds a vector database from a PDF file.

    Args:
        filename (str): Path to the PDF file.
    """
    doc = PyMuPDFLoader(filename).load()

    if CFG.PROPOSITIONIZE:
        texts = propositionize(doc)
    else:
        texts = text_split(doc, CFG.CHUNK_SIZE, CFG.CHUNK_OVERLAP)

    embeddings = build_base_embeddings()
    if CFG.VECTORDB_TYPE == "faiss":
        _build_faiss(texts, embeddings)
    elif CFG.VECTORDB_TYPE == "chroma":
        _build_chroma(texts, embeddings)
    else:
        raise NotImplementedError


def text_split(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> Sequence[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CFG.SEPARATORS,
        length_function=len,
    )
    return text_splitter.split_documents(doc)


def propositionize(doc: Document) -> Sequence[Document]:
    from src.elements.propositionizer import Propositionizer

    propositionizer = Propositionizer()

    texts = text_split(
        doc,
        CFG.PROPOSITIONIZER_CONFIG.CHUNK_SIZE,
        CFG.PROPOSITIONIZER_CONFIG.CHUNK_OVERLAP,
    )

    prop_texts = propositionizer.batch(texts)
    return prop_texts


def _build_faiss(texts, embeddings):
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(CFG.VECTORDB_PATH)


def _build_chroma(texts, embeddings):
    _ = Chroma.from_documents(texts, embeddings, persist_directory=CFG.VECTORDB_PATH)


def load_faiss(embeddings):
    return FAISS.load_local(CFG.VECTORDB_PATH, embeddings)


def load_chroma(embeddings):
    return Chroma(persist_directory=CFG.VECTORDB_PATH, embedding_function=embeddings)
