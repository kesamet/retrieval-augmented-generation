"""
Embeddings
"""
import os

from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from src import CFG

if CFG.PROMPT_TYPE == "llama":
    HYDE_TEMPLATE = """<s>[INST] <<SYS>> Please answer the user's question about the document. <</SYS>>
Question: {question}
Answer: [/INST]"""
elif CFG.PROMPT_TYPE == "mistral":
    HYDE_TEMPLATE = """<s>[INST] Please answer the user's question about the document.
Question: {question}
Answer: [/INST]"""
else:
    raise NotImplementedError


def build_base_embeddings():
    """Builds base embeddings defined in config."""
    base_embeddings = HuggingFaceEmbeddings(
        model_name=os.path.join(CFG.MODELS_DIR, CFG.EMBEDDINGS_MODEL_PATH),
        model_kwargs={"device": CFG.DEVICE},
    )
    return base_embeddings


def build_hyde_embeddings(llm, base_embeddings):
    """Builds hypothetical document embeddings."""
    prompt = PromptTemplate.from_template(HYDE_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings
