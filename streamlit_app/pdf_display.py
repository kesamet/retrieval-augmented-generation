import base64
import os
import re
import tempfile
from difflib import SequenceMatcher
from typing import List, Tuple

import fitz
import streamlit as st


def get_doc_highlighted(filename: str, phrase: str) -> Tuple[fitz.Document, list]:
    """Gets the highlighted document and the page numbers.

    Args:
        filename (str): The path to the PDF file.
        phrase (str): The phrase to search for.

    Returns:
        Tuple[fitz.Document, list]
    """
    doc = fitz.Document(filename)

    all_instances = search_for_phrase(doc, phrase)

    if all_instances:
        doc_highlighted = highlight_phrase(filename, all_instances)
        page_nums = list(all_instances.keys())
        return doc_highlighted, page_nums
    return None, None


def search_for_phrase(
    doc: fitz.Document,
    phrase: str,
    page_nums: list = None,
    cutoff: float = 0.8,
) -> List[fitz.Rect]:
    """Search for a phrase in a PDF document.

    Args:
        doc (fitz.Document): The PDF document to search.
        phrase (str): The phrase to search for.
        page_nums (list): A list of page numbers to search. If None, all pages are searched.
        cutoff (float): The minimum similarity score required to return a match.

    Returns:
        A list of rectangles representing the bounding boxes of the matches.
    """
    if page_nums is None:
        page_nums = range(len(doc))

    cleaned_phrase = re.sub(r"\W+", "", phrase)

    all_instances = dict()
    for i in page_nums:
        if cleaned_phrase not in re.sub(r"\W+", "", doc[i].get_text("text")):
            continue

        instances = list()
        for x0, x1, y0, y1, text, _, _ in doc[i].get_text("blocks"):
            cleaned_text = re.sub(r"\W+", "", text)
            if len(cleaned_text) == 0:
                continue
            slen = (
                SequenceMatcher(None, cleaned_phrase, cleaned_text)
                .find_longest_match()
                .size
            )
            if slen / len(cleaned_phrase) > cutoff or slen / len(cleaned_text) > cutoff:
                instances.append(fitz.Rect(x0, x1, y0, y1))

        if instances:
            all_instances[i] = instances
    return all_instances


def highlight_phrase(filename: str, all_instances: List[fitz.Rect]) -> fitz.Document:
    """Highlights a phrase in a PDF document.

    Args:
        filename (str): The path to the PDF document.
        all_instances (List[fitz.Rect]): A list of rectangles that represent the
            locations of the phrase in the document.

    Returns:
        fitz.Document: The highlighted PDF document.
    """
    doc = fitz.Document(filename)
    for page_num, rects in all_instances.items():
        for rect in rects:
            doc[page_num].add_highlight_annot(rect)
    return doc


def display_pdf(extracted_doc: fitz.Document, page_num: int = 1) -> None:
    """Displays a PDF page in a new window.

    Args:
        extracted_doc (fitz.Document): The PDF document to display.
        page_num (int): The page number to display.
    """
    fh, temp_filename = tempfile.mkstemp()
    try:
        extracted_doc.save(temp_filename, garbage=4, deflate=True, clean=True)
        with open(temp_filename, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" \
width="100%" height="950" type="application/pdf"></iframe>"""
    finally:
        os.close(fh)
        os.remove(temp_filename)

    st.markdown(pdf_display, unsafe_allow_html=True)
