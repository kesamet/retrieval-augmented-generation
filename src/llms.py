"""
LLM
"""
import os

from langchain.llms.ctransformers import CTransformers

from src import CFG


def build_llm():
    """Builds language model defined in config."""
    llm = CTransformers(
        model=os.path.join(CFG.MODELS_DIR, CFG.LLM_MODEL),
        model_type=CFG.LLM_MODEL_TYPE,
        config={
            "max_new_tokens": CFG.MAX_NEW_TOKENS,
            "temperature": CFG.TEMPERATURE,
            "repetition_penalty": CFG.REPETITION_PENALTY,
            "context_length": CFG.CONTEXT_LENGTH,
        },
    )
    return llm
