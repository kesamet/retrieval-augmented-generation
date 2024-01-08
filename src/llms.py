"""
LLM
"""
import os

from langchain.llms.ctransformers import CTransformers
from langchain.callbacks import StreamingStdOutCallbackHandler

from src import CFG


def build_llm(debug: bool = False):
    """Builds language model defined in config."""
    llm = CTransformers(
        model=os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
        model_type=CFG.LLM_TYPE,
        config={
            "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
            "temperature": CFG.LLM_CONFIG.TEMPERATURE,
            "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
            "context_length": CFG.LLM_CONFIG.CONTEXT_LENGTH,
        },
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
    )
    return llm
