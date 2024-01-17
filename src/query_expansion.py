from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableSequence

from src.prompt_templates import MULTI_QUERIES_TEMPLATE


def build_llm_chain(llm: LLM, template: str) -> RunnableSequence:
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain


def build_generated_result_expansion_chain(llm: LLM) -> RunnableSequence:
    template = """<s>[INST] You are a helpful assistant. \
Provide an example answer to the given question, that might be found in a document."
Question: {question}
Output: [/INST]"""

    chain = {"question": RunnablePassthrough()} | build_llm_chain(llm, template)
    return chain


def build_multiple_queries_expansion_chain(llm: LLM) -> RunnableSequence:
    chain = {"question": RunnablePassthrough()} | build_llm_chain(
        llm, MULTI_QUERIES_TEMPLATE
    )
    return chain
