from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableSequence


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
    template = """<s>[INST] You are a helpful assistant. \
Your users are asking questions about documents. \
Suggest up to three additional related questions to help them find the information they need, \
for the provided question. \
Suggest only short questions without compound sentences. \
Suggest a variety of questions that cover different aspects of the topic. \
Make sure they are complete questions, and that they are related to the original question. \
Output one question per line and without numbering.
Question: {question}
Output: [/INST]"""

    chain = {"question": RunnablePassthrough()} | build_llm_chain(llm, template)
    return chain
