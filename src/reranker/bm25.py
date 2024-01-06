from typing import List

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from langchain.schema import Document

from src.reranker.base import BaseReranker

TOKENIZER_NAME = "gpt2"


class BM25Reranker(BaseReranker):
    """Reranker based on BM25."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def rerank(self, query: str, passages: List[Document]) -> List[Document]:
        contents: List[str] = [passage.page_content for passage in passages]

        # tokenize content for bm25 instance
        tokenized_content = self.tokenizer(contents).input_ids

        # tokenize query
        tokenized_query = self.tokenizer([query]).input_ids[0]

        bm25 = BM25Okapi(tokenized_content)
        scores = bm25.get_scores(tokenized_query)

        sorted_pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        sorted_passages = [passage for passage, _ in sorted_pairs]
        return sorted_passages
