"""
Collection of useful tools
"""

from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Web Search Tool
_search = TavilySearchAPIWrapper()
description = (
    "A search engine optimized for comprehensive, accurate, "
    "and trusted results. Useful for when you need to answer questions "
    "about current events or about recent information. "
    "Input should be a search query. "
    "If the user is asking about something that you don't know about, "
    "you should probably use this tool to see if that can provide any information."
)
tavily_tool = TavilySearchResults(api_wrapper=_search, description=description)

# Wikipedia Tool
_wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=_wikipedia.run,
    description=(
        "A wrapper around Wikipedia. Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    ),
)
