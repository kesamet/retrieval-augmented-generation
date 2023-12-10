"""
Collection of useful tools
"""
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool, Tool
from langchain.utilities.wikipedia import WikipediaAPIWrapper


# Knowledge Base Tool
class KnowledgeBaseQueryRun(BaseTool):
    """Tool that searches the Knowledge Base."""

    name: str = "Knowledge Base"
    description: str = "Useful for when you need to answer questions about information from vector database."
    retrieval_qa: RetrievalQA

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.retrieval_qa.run(query)


# Wikipedia Tool
_wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=_wikipedia.run,
    description=(
        "A wrapper around Wikipedia. Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects."
    ),
)
