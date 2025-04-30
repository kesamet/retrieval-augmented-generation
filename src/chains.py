"""
Chains
"""

from typing import Callable, Optional

from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableBranch,
)
from pydantic import BaseModel, Field

from src.prompt_templates import prompts


def create_llm_chain(llm: LLM, system_prompt: str, user_prompt: str) -> Runnable:
    """Creates a generic llm chain."""
    prompt = PromptTemplate.from_template(
        prompts.chat_format.format(system=system_prompt, user=user_prompt)
    )
    llm_chain = prompt | llm | StrOutputParser()
    return llm_chain


def create_condense_question_chain(llm: LLM) -> Runnable:
    """Creates a chain that condenses question and chat history to create a standalone question."""
    condense_question_prompt = PromptTemplate.from_template(prompts.condense_question)
    condense_question_chain = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input
            (lambda x: x["question"]),
        ),
        # If chat history, then we pass inputs to LLM chain
        condense_question_prompt | llm | StrOutputParser(),
    ).with_config(name="condense_question_chain")
    return condense_question_chain


def create_question_answer_chain(llm: LLM) -> Runnable:
    """Creates a question-answer chain.
    Copied from langchain.chains.combine_documents.stuff.create_stuff_documents_chain
    """
    qa_prompt = PromptTemplate.from_template(prompts.qa)

    def format_docs(inputs: dict) -> str:
        return "\n\n".join(doc.page_content for doc in inputs["context"])

    question_answer_chain = (
        RunnablePassthrough.assign(context=format_docs).with_config(run_name="format_inputs")
        | qa_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")
    return question_answer_chain


def create_guardrail_chain(
    llm: LLM, template: str, retry_func: Optional[Callable] = None
) -> Runnable:
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = llm.with_structured_output(grade)
    if retry_func is not None:
        llm_with_tool = retry_func(llm_with_tool)
    prompt = PromptTemplate(template=template)
    chain = (prompt | llm_with_tool).with_config(run_name="guardrail_chain")
    return chain
