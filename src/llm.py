"""
LLM
"""
from langchain.llms import CTransformers

from src import CFG


def build_llm():
    llm = CTransformers(
        model=CFG.LLM_MODEL,
        model_type=CFG.LLM_MODEL_TYPE,
        config={
            "max_new_tokens": CFG.MAX_NEW_TOKENS,
            "temperature": CFG.TEMPERATURE,
        },
    )
    return llm
