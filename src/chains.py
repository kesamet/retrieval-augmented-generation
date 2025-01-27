"""
Chains
"""

from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableBranch,
)

from src.prompt_templates import prompts


def build_condense_question_chain(llm: LLM) -> Runnable:
    """Builds a chain that condenses question and chat history to create a standalone question."""
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
    )
    return condense_question_chain


def build_question_answer_chain(llm: LLM) -> Runnable:
    """Builds a question-answer chain.
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
