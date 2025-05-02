"""
Embeddings
"""

import os

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from src import CFG
from src.prompt_templates import HYDE_TEMPLATE


def load_base_embeddings():
    """Loads base embeddings defined in config."""
    match CFG.EMBEDDINGS_PROVIDER:
        case "huggingface":
            return HuggingFaceEmbeddings(
                model_name=os.path.join(CFG.MODELS_DIR, CFG.EMBEDDINGS),
                model_kwargs={"device": CFG.DEVICE},
            )
        case _:
            raise NotImplementedError(f"{CFG.EMBEDDINGS} not implemented")


def create_hyde_embeddings(llm, base_embeddings):
    """Create hypothetical document embeddings."""
    prompt = PromptTemplate.from_template(HYDE_TEMPLATE)
    llm_chain = prompt | llm

    embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=base_embeddings)
    return embeddings


def load_sagemaker_endpoint_embeddings():
    import boto3
    from langchain_community.embeddings import SagemakerEndpointEmbeddings
    from src.sagemaker_endpoint.embeddings import ContentHandler

    runtime_client = boto3.client("sagemaker-runtime", region_name=CFG.REGION_NAME)
    return SagemakerEndpointEmbeddings(
        endpoint_name=CFG.EMBEDDINGS_ENDPOINT,
        client=runtime_client,
        content_handler=ContentHandler(),
    )
