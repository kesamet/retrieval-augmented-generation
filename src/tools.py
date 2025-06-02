from functools import partial
from typing import Literal, Optional

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    aformat_document,
    format_document,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import RetrieverInput, Tool, tool
from langchain_community.tools import TavilySearchResults

# Web Search Tool
tavily_tool = TavilySearchResults(
    description=(
        "A search engine optimized for comprehensive, accurate, "
        "and trusted results. Useful for when you need to answer questions "
        "about current events or about recent information. "
        "Input should be a search query. "
        "If the user is asking about something that cannot be found in the document, "
        "you should probably use this tool."
    ),
    max_results=4,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)


@tool
def think(thought: str):
    """
    Use the tool to think about something. It will not obtain new information,
    but just append the thought to the log.
    Use it when complex reasoning or some cache memory is needed.

    Args:
        thought (str): A thought to think about.
    """
    pass


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> Tool:
    r"""Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.
        document_prompt: The prompt to use for the document. Defaults to None.
        document_separator: The separator to use between documents. Defaults to "\n\n".
        response_format: The tool response format. If "content" then the output of
            the tool is interpreted as the contents of a ToolMessage. If
            "content_and_artifact" then the output is expected to be a two-tuple
            corresponding to the (content, artifact) of a ToolMessage (artifact
            being a list of documents in this case). Defaults to "content".

    Returns:
        Tool class to pass to an agent.
    """
    document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    func = partial(
        _get_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        response_format=response_format,
    )
    afunc = partial(
        _aget_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
        response_format=response_format,
    )
    return Tool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
        response_format=response_format,
    )


def _get_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks | None = None,
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> str | tuple[str, list[Document]]:
    docs = retriever.invoke(query, config={"callbacks": callbacks})
    content = document_separator.join(format_document(doc, document_prompt) for doc in docs)
    if response_format == "content_and_artifact":
        return (content, docs)
    return content


async def _aget_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks | None = None,
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> str | tuple[str, list[Document]]:
    docs = await retriever.ainvoke(query, config={"callbacks": callbacks})
    content = document_separator.join(
        [await aformat_document(doc, document_prompt) for doc in docs]
    )
    if response_format == "content_and_artifact":
        return (content, docs)
    return content
