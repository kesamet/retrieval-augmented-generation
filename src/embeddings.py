"""
Embeddings
"""

import os

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from src import CFG
from src.prompt_templates import HYDE_TEMPLATE


def build_base_embeddings():
    """Builds base embeddings defined in config."""
    base_embeddings = HuggingFaceEmbeddings(
        model_name=os.path.join(CFG.MODELS_DIR, CFG.EMBEDDINGS_PATH),
        model_kwargs={"device": CFG.DEVICE},
    )
    return base_embeddings


def build_hyde_embeddings(llm, base_embeddings):
    """Builds hypothetical document embeddings."""
    prompt = PromptTemplate.from_template(HYDE_TEMPLATE)
    llm_chain = prompt | llm

    embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=base_embeddings)
    return embeddings
