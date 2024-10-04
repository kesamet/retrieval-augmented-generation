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
