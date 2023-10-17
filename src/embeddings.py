"""
Embeddings
"""
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

from src import CFG
from src.llm import build_llm


def build_base_embeddings() -> HuggingFaceEmbeddings:
    """Builds base embeddings defined in config."""
    base_embeddings = HuggingFaceEmbeddings(
        model_name=CFG.EMBEDDINGS_MODEL,
        model_kwargs={"device": CFG.DEVICE},
    )
    return base_embeddings


def build_hyde_embeddings(llm, base_embeddings):
    """Builds hypothetical document embeddings."""
    hyde_template = """<s>[INST] <<SYS>> Please answer the user's question about the document. <</SYS>>
Question: {question}
Answer:[/INST]"""
    prompt = PromptTemplate(input_variables=["question"], template=hyde_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings
