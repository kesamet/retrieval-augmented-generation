"""
VectorDB
"""
import os
from typing import List

import torch
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src import CFG
from src.embeddings import build_base_embeddings


def build_vectordb(filename: str) -> None:
    """Builds a vector database from a PDF file.

    Args:
        filename (str): Path to the PDF file.
    """
    doc = PyMuPDFLoader(filename).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.CHUNK_SIZE,
        chunk_overlap=CFG.CHUNK_OVERLAP,
        separators=CFG.SEPARATORS,
        length_function=len,
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


class Propositionizer:
    """Based on https://github.com/chentong0/factoid-wiki."""

    def __init__(self):
        model_name = os.path.join(CFG.MODELS_DIR, "./models/propositionizer-wiki-flan-t5-large")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, title: str, section: str, passages: List[Document]) -> List[Document]:
        input_text: List[str] = [
            f"Title: {title}. Section: {section}. Content: {passage.page_content}" for passage in passages
        ]

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(CFG.DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=512).cpu()

        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return output_text
