import json
import os
from typing import List

import torch
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src import CFG


class Propositionizer:
    """Based on https://github.com/chentong0/factoid-wiki."""

    def __init__(self):
        model_name = os.path.join(
            CFG.MODELS_DIR, "./models/propositionizer-wiki-flan-t5-large"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(CFG.DEVICE)
        self.model.eval()

    def generate(self, title: str, section: str, passage: Document) -> List[Document]:
        input_text = (
            f"Title: {title}. Section: {section}. Content: {passage.page_content}"
        )

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            CFG.DEVICE
        )
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=512).cpu()

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        metadata = passage.metadata.copy()
        return [
            Document(page_content=x, metadata=metadata) for x in json.loads(output_text)
        ]

    def batch(
        self, title: str, section: str, passages: List[Document]
    ) -> List[Document]:
        pass
