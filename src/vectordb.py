"""
Build vectordb
"""
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

from src import CFG


def build_vectordb():
    loader = DirectoryLoader(
        CFG.DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.CHUNK_SIZE, chunk_overlap=CFG.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=CFG.EMBEDDINGS,
        model_kwargs={"device": CFG.DEVICE},
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(CFG.DB_FAISS_PATH)


if __name__ == "__main__":
    build_vectordb()
