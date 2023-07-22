"""
LLM
"""
from langchain.llms import CTransformers

from src import CFG


def build_llm():
    llm = CTransformers(
        model=CFG.MODEL_BIN_PATH,
        model_type=CFG.MODEL_TYPE,
        config={
            "max_new_tokens": CFG.MAX_NEW_TOKENS,
            "temperature": CFG.TEMPERATURE,
        },
    )
    return llm
