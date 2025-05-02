"""
LLM
"""

import os
from typing import Dict

from langchain_core.callbacks import StreamingStdOutCallbackHandler

from src import CFG

DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_CONTEXT_LENGTH = 4000


def load_llm():
    """Loads LLM defined in config."""
    match CFG.LLM_PROVIDER:
        case "llamacpp":
            return load_llamacpp(
                os.path.join(CFG.MODELS_DIR, CFG.LLM),
                config={
                    "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repeat_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                    "n_ctx": CFG.LLM_CONFIG.CONTEXT_LENGTH,
                },
            )
        case "groq":
            return load_chatgroq(
                model_name=CFG.LLM,
                config={
                    "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                },
            )
        case "ollama":
            return load_chatollama(
                CFG.LLM,
                config={
                    "num_predict": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                    "repeat_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                },
            )
        case "openai":
            return load_chatopenai(
                CFG.LLM,
                config={
                    "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                },
            )
        case "google":
            return load_chatgooglegenerativeai(
                CFG.LLM,
                config={
                    "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                    "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                },
            )
        case _:
            raise NotImplementedError(f"{CFG.LLM} not implemented")


def load_llamacpp(model_path: str, config: dict | None = None, debug: bool = False, **kwargs):
    """Builds LLM using LlamaCpp."""
    from langchain_community.llms.llamacpp import LlamaCpp

    if config is None:
        config = {
            "max_tokens": DEFAULT_MAX_NEW_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "repeat_penalty": DEFAULT_REPETITION_PENALTY,
            "n_ctx": DEFAULT_CONTEXT_LENGTH,
        }

    llm = LlamaCpp(
        model_path=model_path,
        **config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm


def load_chatgroq(model_name: str = "mixtral-8x7b-32768", config: dict | None = None, **kwargs):
    """For Groq LLM."""
    from langchain_groq import ChatGroq

    if config is None:
        config = {
            "max_tokens": DEFAULT_MAX_NEW_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
        }

    llm = ChatGroq(
        model_name=model_name,
        **config,
        streaming=True,
        **kwargs,
    )
    return llm


def load_chatollama(model: str, config: dict | None = None, debug: bool = False, **kwargs):
    """For LLM deployed as an API."""
    from langchain_ollama import ChatOllama

    if config is None:
        config = {
            "num_predict": DEFAULT_MAX_NEW_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "repeat_penalty": DEFAULT_REPETITION_PENALTY,
        }

    llm = ChatOllama(
        model=model,
        **config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm


def load_chatopenai(openai_api_base: str, config: dict | None = None, **kwargs):
    """For LLM deployed as an OpenAI compatible API."""
    from langchain_openai import ChatOpenAI

    if config is None:
        config = {
            "max_tokens": DEFAULT_MAX_NEW_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
        }

    llm = ChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_key="sk-xxx",
        **config,
        streaming=True,
        **kwargs,
    )
    return llm


def load_chatgooglegenerativeai(
    model: str = "gemini-1.5-flash", config: dict | None = None, **kwargs
):
    from langchain_google_genai import ChatGoogleGenerativeAI

    if config is None:
        config = {
            "temperature": DEFAULT_TEMPERATURE,
        }

    llm = ChatGoogleGenerativeAI(model=model, **config, **kwargs)
    return llm


def load_sagemaker_endpoint_llm(model_kwargs: Dict | None = None, **kwargs):
    import boto3
    from src.sagemaker_endpoint.llm import SagemakerEndpointLLM, ContentHandler

    if model_kwargs is None:
        model_kwargs = {
            "temperature": CFG.LLM_CONFIG.TEMPERATURE,
            "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
        }

    runtime_client = boto3.client("sagemaker-runtime", region_name=CFG.REGION_NAME)
    return SagemakerEndpointLLM(
        endpoint_name=CFG.LLM_ENDPOINT,
        client=runtime_client,
        content_handler=ContentHandler(),
        model_kwargs=model_kwargs,
        **kwargs,
    )
