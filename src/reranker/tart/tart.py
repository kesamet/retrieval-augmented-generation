import os
from typing import List

import torch
import torch.nn.functional as F
from langchain.schema import Document

from src import CFG
from src.reranker.base import BaseReranker
from .modeling_enc_t5 import EncT5ForSequenceClassification
from .tokenization_enc_t5 import EncT5Tokenizer


class TARTReranker(BaseReranker):
    """Reranker based on TART (https://github.com/facebookresearch/tart)."""

    def __init__(self, instruction: str):
        model_path = os.path.join(CFG.MODELS_DIR, "models/tart-full-flan-t5-xl")
        self.tokenizer = EncT5Tokenizer.from_pretrained(model_path)
        self.model = EncT5ForSequenceClassification.from_pretrained(model_path).to(
            CFG.DEVICE
        )
        self.model.eval()
        self.instruct_template = instruction + " [SEP] {query}"

    def rerank(self, query: str, passages: List[Document]) -> List[Document]:
        contents: List[str] = [passage.page_content for passage in passages]
        instruction_queries: List[str] = [
            self.instruct_template.format(query=query) for _ in range(len(contents))
        ]

        features = self.tokenizer(
            instruction_queries,
            contents,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            scores = self.model(**features).logits
            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]

        sorted_pairs = sorted(
            zip(passages, normalized_scores), key=lambda x: x[1], reverse=True
        )
        sorted_passages = [passage for passage, _ in sorted_pairs]
        return sorted_passages
