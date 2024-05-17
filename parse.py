"""
To parse PDFs and save them as vectors in vector database
"""

import argparse
import os

from tqdm import tqdm

from src.embeddings import build_base_embeddings
from src.vectordb import load_pdf, text_split, save_vectordb

BASE_EMBEDDINGS = build_base_embeddings()
VECTORDB_DIR = "./vectordb"
VECTORDB_TYPE = "faiss"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepaths", type=str, nargs=argparse.ONE_OR_MORE)
    args = parser.parse_args()

    for filepath in tqdm(args.filepaths):
        print(f"\nProcessing {filepath}")
        parts = load_pdf(filepath)
        docs = text_split(parts)

        dest = os.path.join(
            VECTORDB_DIR,
            VECTORDB_TYPE + "_" + os.path.splitext(os.path.basename(filepath))[0],
        )
        save_vectordb(docs, BASE_EMBEDDINGS, dest, VECTORDB_TYPE)
