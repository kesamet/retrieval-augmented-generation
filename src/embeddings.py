"""
Embeddings
"""
from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from src import CFG


def build_base_embeddings():
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
