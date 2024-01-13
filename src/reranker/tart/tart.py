import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from langchain.schema import Document
from langchain.pydantic_v1 import Extra
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from src import CFG
from .modeling_enc_t5 import EncT5ForSequenceClassification
from .tokenization_enc_t5 import EncT5Tokenizer


class TARTReranker(BaseDocumentCompressor):
    """Reranker based on TART (https://github.com/facebookresearch/tart)."""

    model_path: str = os.path.join(CFG.MODELS_DIR, CFG.RERANKER_PATH)
    tokenizer = EncT5Tokenizer.from_pretrained(model_path)
    model = EncT5ForSequenceClassification.from_pretrained(model_path)
    """Model to use for reranking."""
    instruction: str = "Find passage to answer given question"
    """Instruction."""
    top_n: int = 4
    """Number of documents to return."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using TART model.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results

    def rerank(self, query: str, docs: Sequence[str]) -> Sequence[Tuple[int, float]]:
        """
        Reranks a list of documents based on a given query using a pre-trained model.

        Args:
            query: The query string.
            docs: The list of documents to be reranked.

        Returns:
            A list of tuples containing the index of the document and its reranked score.
        """
        instruction_queries: Sequence[str] = [
            f"{self.instruction} [SEP] {query}" for _ in range(len(docs))
        ]

        features = self.tokenizer(
            instruction_queries,
            docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            scores = self.model(**features).logits
            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]

        results = sorted(enumerate(normalized_scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]
