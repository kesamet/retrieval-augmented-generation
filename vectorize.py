"""
To parse PDFs and save them as vectors in vector database
"""

import argparse
import os

from tqdm import tqdm
from langchain.schema import Document

from src.embeddings import load_base_embeddings
from src.llms import googlegenerativeai
from src.vectordbs import load_pdf, text_split, save_faiss
from src.parser import get_title
from src.elements.raptor import Raptorizer

EMBEDDING_FUNCTION = load_base_embeddings()
LLM = googlegenerativeai("gemini-1.5-flash")
RAPTORIZER = Raptorizer(EMBEDDING_FUNCTION, LLM, "gemini")

VECTORDB_DIR = "./vectordb"
VECTORDB_TYPE = "faiss"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepaths", type=str, nargs=argparse.ONE_OR_MORE)
    args = parser.parse_args()

    for filepath in tqdm(args.filepaths):
        print(f"\nProcessing {filepath}")
        parts = load_pdf(filepath)
        title = get_title(parts)
        print(f"Title: {title}")

        docs = text_split(parts)

        leaf_texts = [doc.page_content for doc in docs]
        results = RAPTORIZER.recursive_embed_cluster_summarize(
            leaf_texts, title, level=1, n_levels=3
        )

        metadata = docs[0].metadata.copy()
        metadata.pop("page_number", None)

        summarize_docs = []
        for level in sorted(results.keys()):
            summaries = results[level][1]["summaries"].tolist()
            summaries = [Document(page_content=text, metadata=metadata) for text in summaries]
            summarize_docs.extend(summaries)

        docs.extend(summarize_docs)

        dest = os.path.join(
            VECTORDB_DIR,
            VECTORDB_TYPE + "_" + os.path.splitext(os.path.basename(filepath))[0],
        )
        save_faiss(docs, EMBEDDING_FUNCTION, dest)
