"""
Adapted from
https://github.com/shiv248/Streamlit-x-LangGraph-Cookbooks/blob/master/StreamlitCallbackHandler_example/st_callable_util.py
"""

from typing import Callable, TypeVar, List, Optional, Any, Dict
import inspect

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# Define available LLM event types for filtering
AVAILABLE_EVENT_TYPES = [
    # LLM-specific events
    "on_llm_start",  # "Called when LLM starts generating"
    "on_llm_new_token",  # "Called for each new token generated (streaming)"
    "on_llm_end",  # "Called when LLM finishes generating"
    "on_llm_error",  # "Called when LLM encounters an error"
    # Chain events
    "on_chain_start",  # "Called when a chain starts"
    "on_chain_end",  # "Called when a chain ends"
    "on_chain_error",  # "Called when a chain encounters an error"
    # Tool events
    "on_tool_start",  # "Called when a tool starts"
    "on_tool_end",  # "Called when a tool ends"
    "on_tool_error",  # "Called when a tool encounters an error"
    # Agent events
    "on_agent_action",  # "Called when an agent takes an action"
    "on_agent_finish",  # "Called when an agent finishes"
    # Text events
    "on_text",  # "Called for text output"
    # Retriever events
    "on_retriever_start",  # "Called when a retriever starts"
    "on_retriever_end",  # "Called when a retriever ends"
    "on_retriever_error",  # "Called when a retriever encounters an error"
    # Embedding events
    "on_embedding_start",  # "Called when embeddings start"
    "on_embedding_end",  # "Called when embeddings end"
    "on_embedding_error",  # "Called when embeddings encounter an error"
    # Chat model events
    "on_chat_model_start",  # "Called when a chat model starts"
    "on_chat_model_end",  # "Called when a chat model ends"
    "on_chat_model_error",  # "Called when a chat model encounters an error"
    # Retry events
    "on_retry",  # "Called when a retry occurs"
    # Run events
    "on_run_start",  # "Called when a run starts"
    "on_run_end",  # "Called when a run ends"
    "on_run_error",  # "Called when a run encounters an error"
]


class FilteredStreamlitCallbackHandler(StreamlitCallbackHandler):
    """
    Enhanced StreamlitCallbackHandler that supports filtering tool responses.
    """

    def __init__(
        self,
        parent_container: DeltaGenerator,
        show_tool_responses: bool = True,
        tool_response_filter: Optional[Callable[[str, Any], bool]] = None,
        **kwargs,
    ):
        super().__init__(parent_container, **kwargs)
        self.show_tool_responses = show_tool_responses
        self.tool_response_filter = tool_response_filter

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """
        Override to filter tool responses based on configuration.
        """
        if not self.show_tool_responses:
            return

        # Apply custom filter if provided
        if self.tool_response_filter is not None:
            tool_name = kwargs.get("name", "unknown_tool")
            if not self.tool_response_filter(tool_name, output):
                return

        # Call the original method
        super().on_tool_end(output, **kwargs)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """
        Override to filter tool start events based on configuration.
        """
        if not self.show_tool_responses:
            return

        # Apply custom filter if provided
        if self.tool_response_filter is not None:
            tool_name = serialized.get("name", "unknown_tool")
            if not self.tool_response_filter(tool_name, input_str):
                return

        # Call the original method
        super().on_tool_start(serialized, input_str, **kwargs)


def get_streamlit_callback(
    parent_container: DeltaGenerator,
    event_types: Optional[List[str]] = None,
    exclude_event_types: Optional[List[str]] = None,
    show_tool_responses: bool = True,
    tool_response_filter: Optional[Callable[[str, Any], bool]] = None,
) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that integrates fully with any LangChain ChatLLM
    integration, updating the provided Streamlit container with outputs such as tokens,
    model responses, and intermediate steps. This function ensures that all callback methods
    run within the Streamlit execution context, fixing the NoSessionContext() error commonly
    encountered in Streamlit callbacks.

    Args:
        parent_container (DeltaGenerator): The container where the text will be rendered
            during the LLM interaction.
        event_types (Optional[List[str]]): List of specific event types to include.
            If None, all event types are included. Available types:
            - LLM events: on_llm_start, on_llm_new_token, on_llm_end, on_llm_error
            - Chain events: on_chain_start, on_chain_end, on_chain_error
            - Tool events: on_tool_start, on_tool_end, on_tool_error
            - Agent events: on_agent_action, on_agent_finish
            - Text events: on_text
            - Retriever events: on_retriever_start, on_retriever_end, on_retriever_error
            - Embedding events: on_embedding_start, on_embedding_end, on_embedding_error
            - Chat model events: on_chat_model_start, on_chat_model_end, on_chat_model_error
            - Retry events: on_retry
            - Run events: on_run_start, on_run_end, on_run_error
        exclude_event_types (Optional[List[str]]): List of specific event types to exclude.
            Takes precedence over event_types if both are provided.
        show_tool_responses (bool): Whether to show tool calling responses. Default is True.
        tool_response_filter (Optional[Callable[[str, Any], bool]]): Custom filter function
            for tool responses. Takes tool name and response/output as arguments and returns
            True if the response should be shown, False otherwise.

    Returns:
        BaseCallbackHandler: An instance of StreamlitCallbackHandler configured for full
            integration with ChatLLM, enabling dynamic updates in the Streamlit app.

    Examples:
        # Include only LLM events
        callback = get_streamlit_callback(
            st.empty(),
            event_types=["on_llm_start", "on_llm_new_token", "on_llm_end"]
        )

        # Exclude specific events
        callback = get_streamlit_callback(
            st.empty(),
            exclude_event_types=["on_llm_error", "on_chain_error"]
        )

        # Include only streaming events
        callback = get_streamlit_callback(
            st.empty(),
            event_types=["on_llm_new_token"]
        )

        # Hide all tool responses
        callback = get_streamlit_callback(
            st.empty(),
            show_tool_responses=False
        )

        # Filter tool responses by tool name
        def filter_tools(tool_name: str, response: Any) -> bool:
            # Only show responses from specific tools
            allowed_tools = ["search", "calculator"]
            return tool_name in allowed_tools

        callback = get_streamlit_callback(
            st.empty(),
            tool_response_filter=filter_tools
        )

        # Filter tool responses by content
        def filter_by_content(tool_name: str, response: Any) -> bool:
            # Hide responses that are too long or contain sensitive info
            if isinstance(response, str) and len(response) > 1000:
                return False
            if isinstance(response, str) and "error" in response.lower():
                return False
            return True

        callback = get_streamlit_callback(
            st.empty(),
            tool_response_filter=filter_by_content
        )
    """

    # Define a type variable for generic type hinting in the decorator, ensuring the original
    # function and wrapped function maintain the same return type.
    fn_return_type = TypeVar("fn_return_type")

    # Decorator function to add Streamlit's execution context to a function
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the execution context.
        This is necessary for interacting with Streamlit components from within callback functions
        and prevents the NoSessionContext() error by adding the correct session context.

        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated,
                typically a callback method.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the context setup.
        """
        # Retrieve the current Streamlit script execution context.
        # This context holds session information necessary for Streamlit operations.
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            If the Streamlit context is not set, it can lead to NoSessionContext() errors,
            which this wrapper resolves by ensuring that the correct context is used
            when the function runs.

            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            # Add the previously captured Streamlit context to the current execution.
            # This step fixes NoSessionContext() errors by ensuring that Streamlit knows which
            # session is executing the code, allowing it to properly manage session state
            # and updates.
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of our enhanced StreamlitCallbackHandler with the provided container
    st_cb = FilteredStreamlitCallbackHandler(
        parent_container,
        show_tool_responses=show_tool_responses,
        tool_response_filter=tool_response_filter,
    )

    # Determine which event types to process
    if exclude_event_types:
        # If exclude list is provided, remove those events
        event_types_to_process = AVAILABLE_EVENT_TYPES - set(exclude_event_types)
    elif event_types:
        # If include list is provided, only process those events
        event_types_to_process = set(event_types) & AVAILABLE_EVENT_TYPES
    else:
        # If neither is provided, process all events
        event_types_to_process = AVAILABLE_EVENT_TYPES

    # Iterate over all methods of the StreamlitCallbackHandler instance
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith("on_") and method_name in event_types_to_process:
            # Wrap each callback method with the Streamlit context setup to prevent session errors
            setattr(
                st_cb, method_name, add_streamlit_context(method_func)
            )  # Replace the method with the wrapped version

    # Return the fully configured StreamlitCallbackHandler instance, now context-aware
    # and integrated with any ChatLLM
    return st_cb


# Predefined tool response filters
def create_tool_name_filter(allowed_tools: List[str]) -> Callable[[str, Any], bool]:
    """
    Create a filter that only shows responses from specific tools.

    Args:
        allowed_tools: List of tool names that should show responses

    Returns:
        Callable: Filter function that can be passed to get_streamlit_callback
    """

    def filter_func(tool_name: str, response: Any) -> bool:
        return tool_name in allowed_tools

    return filter_func


def create_tool_content_filter(
    max_length: Optional[int] = None,
    exclude_keywords: Optional[List[str]] = None,
    include_keywords: Optional[List[str]] = None,
) -> Callable[[str, Any], bool]:
    """
    Create a filter based on tool response content.

    Args:
        max_length: Maximum length of response to show (None for no limit)
        exclude_keywords: Keywords that cause response to be hidden
        include_keywords: Keywords that must be present to show response

    Returns:
        Callable: Filter function that can be passed to get_streamlit_callback
    """

    def filter_func(tool_name: str, response: Any) -> bool:
        if not isinstance(response, str):
            return True

        # Check length
        if max_length is not None and len(response) > max_length:
            return False

        # Check exclude keywords
        if exclude_keywords:
            response_lower = response.lower()
            for keyword in exclude_keywords:
                if keyword.lower() in response_lower:
                    return False

        # Check include keywords
        if include_keywords:
            response_lower = response.lower()
            for keyword in include_keywords:
                if keyword.lower() not in response_lower:
                    return False

        return True

    return filter_func


def create_tool_type_filter(tool_types: List[str]) -> Callable[[str, Any], bool]:
    """
    Create a filter based on tool types (e.g., 'search', 'calculator', 'database').

    Args:
        tool_types: List of tool types to show responses for

    Returns:
        Callable: Filter function that can be passed to get_streamlit_callback
    """

    def filter_func(tool_name: str, response: Any) -> bool:
        # Simple heuristic to determine tool type from name
        tool_name_lower = tool_name.lower()
        for tool_type in tool_types:
            if tool_type.lower() in tool_name_lower:
                return True
        return False

    return filter_func
