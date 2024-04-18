"""
LLM
"""

import os

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_openai import ChatOpenAI

from src import CFG


def build_llm():
    """Builds LLM defined in config."""
    if CFG.LLM_PATH.endswith(".gguf"):
        if CFG.USE_CTRANSFORMERS:
            return build_ctransformers(
                os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
                config={
                    "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                    "context_length": CFG.LLM_CONFIG.CONTEXT_LENGTH,
                },
            )
        return build_llamacpp(
            os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
            config={
                "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                "repeat_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                "n_ctx": CFG.LLM_CONFIG.CONTEXT_LENGTH,
            },
        )
    elif CFG.LLM_PATH.startswith("http"):
        return chatopenai(
            CFG.LLM_PATH,
            config={
                "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                "temperature": CFG.LLM_CONFIG.TEMPERATURE,
            },
        )
    else:
        raise NotImplementedError


def build_ctransformers(
    model_path: str, config: dict | None = None, debug: bool = False, **kwargs
):
    """Builds LLM using CTransformers."""
    if config is None:
        config = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "repetition_penalty": 1.1,
            "context_length": 4000,
        }

    llm = CTransformers(
        model=model_path,
        config=config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm


def build_llamacpp(
    model_path: str, config: dict | None = None, debug: bool = False, **kwargs
):
    """Builds LLM using LlamaCpp."""
    if config is None:
        config = {
            "max_tokens": 512,
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "n_ctx": 4000,
        }

    llm = LlamaCpp(
        model_path=model_path,
        **config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm


def chatopenai(openai_api_base: str, config: dict | None = None, **kwargs):
    """For LLM deployed as an API."""
    if config is None:
        config = {
            "max_tokens": 512,
            "temperature": 0.2,
        }

    llm = ChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_key="sk-xxx",
        **config,
        streaming=True,
        **kwargs,
    )
    return llm
