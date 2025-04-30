from langchain_core.language_models import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable

from src.prompt_templates import MULTI_QUERIES_TEMPLATE, GENERATED_RESULT_TEMPLATE


def _create_llm_chain(llm: LLM, template: str) -> Runnable:
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain


def build_multiple_queries_expansion_chain(llm: LLM) -> Runnable:
    chain = {"question": RunnablePassthrough()} | _create_llm_chain(llm, MULTI_QUERIES_TEMPLATE)
    return chain


def build_generated_result_expansion_chain(llm: LLM) -> Runnable:
    chain = {"question": RunnablePassthrough()} | _create_llm_chain(llm, GENERATED_RESULT_TEMPLATE)
    return chain
