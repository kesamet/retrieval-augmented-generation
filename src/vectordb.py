"""
VectorDB
"""

import os
import pickle
import uuid
from typing import Any, Optional, Sequence

from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS

from src import CFG, logger
from src.embeddings import build_base_embeddings


def build_vectordb(filename: str) -> None:
    """Builds a vector database from a PDF file."""
    doc = PyMuPDFLoader(filename).load()
    embedding_function = build_base_embeddings()

    if CFG.TEXT_SPLIT_MODE == "simple":
        docs = simple_text_split(doc, CFG.CHUNK_SIZE, CFG.CHUNK_OVERLAP)
        save_vectorstore(docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
    elif CFG.TEXT_SPLIT_MODE == "propositionize":
        docs = propositionize(doc)
        save_vectorstore(docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE)
    elif CFG.TEXT_SPLIT_MODE == "parent_document":
        child_docs, parents = parent_document_split(doc)
        save_vectorstore(
            child_docs, embedding_function, CFG.VECTORDB_PATH, CFG.VECTORDB_TYPE
        )

        with open(CFG.PARENT_DOCS_PATH, "wb") as handle:
            pickle.dump(parents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise NotImplementedError


def simple_text_split(
    doc: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> Sequence[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CFG.SEPARATORS,
        length_function=len,
    )
    return text_splitter.split_documents(doc)


def parent_document_split(
    doc: Sequence[Document],
) -> tuple[Sequence[Document], tuple[list[str], Sequence[Document]]]:
    """ParentDocumentRetriever"""
    id_key = "doc_id"

    parent_docs = simple_text_split(doc, 2000, 0)
    doc_ids = [str(uuid.uuid4()) for _ in parent_docs]

    child_docs = []
    for i, pdoc in enumerate(parent_docs):
        _sub_docs = simple_text_split([pdoc], 400, 0)
        for _doc in _sub_docs:
            _doc.metadata[id_key] = doc_ids[i]
        child_docs.extend(_sub_docs)
    return child_docs, (doc_ids, parent_docs)


def propositionize(doc: Sequence[Document]) -> Sequence[Document]:
    from src.elements.propositionizer import Propositionizer

    propositionizer = Propositionizer()

    texts = simple_text_split(
        doc,
        CFG.PROPOSITIONIZER_CONFIG.CHUNK_SIZE,
        CFG.PROPOSITIONIZER_CONFIG.CHUNK_OVERLAP,
    )

    prop_texts = propositionizer.batch(texts)
    return prop_texts


def save_vectorstore(
    docs: Sequence[Document],
    embedding_function: Embeddings,
    persist_directory: str,
    vectordb_type: str,
) -> None:
    """Saves a vector database to disk."""
    logger.info(f"persist_directory = {persist_directory}")

    if vectordb_type == "faiss":
        vectorstore = FAISS.from_documents(docs, embedding_function)
        vectorstore.save_local(persist_directory)
    elif vectordb_type == "chroma":
        _ = Chroma.from_documents(
            docs,
            embedding_function,
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )
    else:
        raise NotImplementedError


def load_faiss(
    embedding_function: Embeddings, persist_directory: Optional[str] = None
) -> VectorStore:
    """Loads a FAISS index from disk."""
    if persist_directory is None:
        persist_directory = CFG.VECTORDB_PATH
    logger.info(f"persist_directory = {persist_directory}")

    return FAISS.load_local(
        persist_directory, embedding_function, allow_dangerous_deserialization=True
    )


def load_chroma(
    embedding_function: Embeddings, persist_directory: Optional[str] = None
) -> VectorStore:
    """Loads a Chroma index from disk."""
    if persist_directory is None:
        persist_directory = CFG.VECTORDB_PATH
    if not os.path.exists(persist_directory):
        raise FileNotFoundError
    logger.info(f"persist_directory = {persist_directory}")

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )


def load_pkl(filename: str) -> Any:
    with open(filename, "rb") as handle:
        return pickle.load(handle)
