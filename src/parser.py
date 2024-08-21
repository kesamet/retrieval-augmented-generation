"""
Parser
"""

import uuid
from typing import Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean

from src import CFG


def load_pdf(filename: str) -> Sequence:
    """Loads pdf."""
    # from langchain_community.document_loaders.pdf import PyMuPDFLoader
    # return PyMuPDFLoader(filename).load()
    return partition_pdf(filename, strategy="fast")


def get_title(elements: Sequence[Element]) -> str:
    for element in elements:
        if element.category == "Title":
            return element.text


def text_split(elements: Sequence[Element]) -> Sequence[Document]:
    """Text chunking using unstructured."""
    select_elements = [
        element
        for element in elements
        if element.category in ["Title", "NarrativeText", "ListItem"]
    ]
    chunked_elements = chunk_by_title(select_elements, max_characters=1200, new_after_n_chars=1000)

    documents = []
    for element in chunked_elements:
        text = clean(element.text, extra_whitespace=True, bullets=True)

        x = element.metadata.to_dict()
        metadata = {
            "file_directory": x["file_directory"],
            "source": x["filename"],
            "last_modified": x["last_modified"],
            "page_number": x["page_number"],
            "filetype": x["filetype"],
        }

        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def simple_text_split(
    docs: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> Sequence[Document]:
    """Text chunking using langchain RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    texts = text_splitter.split_documents(docs)

    # MuPDF pages use 0-based indexing. Set page_number = page + 1
    for t in texts:
        t.metadata["page_number"] = t.metadata["page"] + 1
    return texts


def parent_document_split(
    docs: Sequence[Document],
) -> tuple[Sequence[Document], tuple[list[str], Sequence[Document]]]:
    """Text chunking for ParentDocumentRetriever."""
    id_key = "doc_id"

    parent_docs = simple_text_split(docs, 2000, 0)
    doc_ids = [str(uuid.uuid4()) for _ in parent_docs]

    child_docs = []
    for i, pdoc in enumerate(parent_docs):
        _sub_docs = simple_text_split([pdoc], 400, 0)
        for _doc in _sub_docs:
            _doc.metadata[id_key] = doc_ids[i]
        child_docs.extend(_sub_docs)
    return child_docs, (doc_ids, parent_docs)


def propositionize(docs: Sequence[Document]) -> Sequence[Document]:
    """Text chunking with Propositionizer."""
    from src.elements.propositionizer import Propositionizer

    propositionizer = Propositionizer()

    texts = simple_text_split(
        docs,
        CFG.PROPOSITIONIZER_CONFIG.CHUNK_SIZE,
        CFG.PROPOSITIONIZER_CONFIG.CHUNK_OVERLAP,
    )

    prop_texts = propositionizer.batch(texts)
    return prop_texts


def raptorize(docs: Sequence[Document], title: str) -> Sequence[Document]:
    from src.embeddings import build_base_embeddings
    from src.llms import googlegenerativeai
    from src.elements.raptor import Raptorizer

    base_embeddings = build_base_embeddings()
    llm = googlegenerativeai("gemini-1.5-flash")
    raptorizer = Raptorizer(base_embeddings, llm, "gemini")

    leaf_texts = [doc.page_content for doc in docs]
    results = raptorizer.recursive_embed_cluster_summarize(leaf_texts, title, level=1, n_levels=3)

    # Flatten all summaries and add them to docs
    metadata = docs[0].metadata
    summarize_docs = []
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        summaries = [Document(page_content=text, metadata=metadata) for text in summaries]
        summarize_docs.extend(summaries)
    return docs + summarize_docs
