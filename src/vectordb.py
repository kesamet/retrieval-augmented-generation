"""
Build vectordb
"""
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from src import CFG


def build_vectordb(filename: str):
    documents = PyMuPDFLoader(filename).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.CHUNK_SIZE, chunk_overlap=CFG.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=CFG.EMBEDDINGS_MODEL,
        model_kwargs={"device": CFG.DEVICE},
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(CFG.VECTORDB_FAISS_PATH)


if __name__ == "__main__":
    build_vectordb()
