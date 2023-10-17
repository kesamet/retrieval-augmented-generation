"""
LLM
"""
from langchain.llms import CTransformers

from src import CFG


def build_llm():
    """Builds language model defined in config."""
    llm = CTransformers(
        model=CFG.LLM_MODEL,
        model_type=CFG.LLM_MODEL_TYPE,
        config={
            "max_new_tokens": CFG.MAX_NEW_TOKENS,
            "temperature": CFG.TEMPERATURE,
            "repetition_penalty": CFG.REPETITION_PENALTY,
        },
    )
    return llm
