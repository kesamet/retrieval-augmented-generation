from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
    # ReActSingleInputOutputParser,
)

from src.prompt_templates import (
    REACT_JSON_TEMPLATE,
    # REACT_TEMPLATE,
)


def build_agent_executor(llm, tools, chat_format):
    prompt = PromptTemplate.from_template(
        chat_format.format(system="You are a helpful assistant.", user=REACT_JSON_TEMPLATE)
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActJsonSingleInputOutputParser(),
        # output_parser=ReActSingleInputOutputParser(),
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=2,
    )
    return agent_executor
