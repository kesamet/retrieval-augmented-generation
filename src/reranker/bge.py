import os
from typing import List

import torch
from langchain.schema import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import CFG
from src.reranker.base import BaseReranker


class BGEReranker(BaseReranker):
    """Reranker based on BGE-reranker (https://huggingface.co/BAAI/bge-reranker-base)."""

    def __init__(self):
        model_path = os.path.join(CFG.MODELS_DIR, "models/bge-reranker-base")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            CFG.DEVICE
        )
        self.model.eval()

    def rerank(self, query: str, passages: List[Document]) -> List[Document]:
        pairs: List[str] = [[query, passage.page_content] for passage in passages]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
                .numpy()
            )

        sorted_pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        sorted_passages = [passage for passage, _ in sorted_pairs]
        return sorted_passages
