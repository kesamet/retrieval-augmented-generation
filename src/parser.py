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


def read_pdf(filename: str) -> Sequence:
    """Read pdf."""
    # from langchain_community.document_loaders.pdf import PyMuPDFLoader
    # return PyMuPDFLoader(filename).load()
    return partition_pdf(filename, strategy="fast")


def text_split(elements: Sequence[Element]) -> Sequence[Document]:
    """Text split."""
    narrative_elements = [
        element for element in elements if element.category == "NarrativeText"
    ]
    narrative_elements = chunk_by_title(
        narrative_elements, max_characters=2000, new_after_n_chars=1500
    )

    documents = []
    for element in narrative_elements:
        text = clean(element.text, extra_whitespace=True)

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
    doc: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> Sequence[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
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
