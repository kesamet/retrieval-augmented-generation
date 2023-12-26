import json
import os
from typing import List, Union

import torch
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset

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

    def _predict(self, texts: Union[str, List[str]]) -> List[str]:
        input_ids = self.tokenizer(texts, return_tensors="pt").input_ids.to(CFG.DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=512).cpu()

        output_texts = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return output_texts

    def generate(
        self, passage: Document, title: str = "", section: str = ""
    ) -> List[Document]:
        input_text = (
            f"Title: {title}. Section: {section}. Content: {passage.page_content}"
        )
        output_text = self._predict(input_text)[0]
        metadata = passage.metadata.copy()
        return [
            Document(page_content=x, metadata=metadata) for x in json.loads(output_text)
        ]

    def batch(
        self,
        passages: List[Document],
        title: str = "",
        section: str = "",
    ) -> List[Document]:
        data_set = DocDataset(passages, title=title, section=section)
        data_loader = DataLoader(
            data_set, batch_size=16, shuffle=False, drop_last=False
        )
        prop_texts = []
        for data in data_loader:
            input_texts, sources = data
            output_texts = self._predict(input_texts)

            for output_text, source, input_text in zip(
                output_texts, sources, input_texts
            ):
                try:
                    prop_texts.extend(
                        [
                            Document(page_content=x, metadata={"source": source})
                            for x in json.loads(output_text)
                        ]
                    )
                except:
                    prop_texts.append(
                        Document(page_content=input_text, metadata={"source": source})
                    )
        return prop_texts


class DocDataset(Dataset):
    def __init__(self, passages: List[Document], title: str = "", section: str = ""):
        self.texts = [
            f"Title: {title}. Section: {section}. Content: {passage.page_content}"
            for passage in passages
        ]
        self.sources = [passage.metadata["source"] for passage in passages]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.sources[idx]
