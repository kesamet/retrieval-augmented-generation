import json
from typing import Any, Callable, Dict, Literal, List, Optional, Sequence, Type, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_community.llms.sagemaker_endpoint import SagemakerEndpoint, LLMContentHandler, LineIterator
from langchain_community.llms.utils import enforce_stop_tokens


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict, stream: bool = False) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs, "stream": stream})
        return input_str.encode("utf-8")
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_text"]
    

class SagemakerEndpointLLM(SagemakerEndpoint):
    """Adapted from SagemakerEndpoint to fix streaming."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(prompt, _model_kwargs, self.streaming)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        if self.streaming and run_manager:
            try:
                resp = self.client.invoke_endpoint_with_response_stream(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=self.content_handler.content_type,
                    **_endpoint_kwargs,
                )
                iterator = LineIterator(resp["Body"])
                current_completion: str = ""
                start_json = b"{"
                for line in iterator:
                    if line != b"" and start_json in line:
                        resp_output = json.loads(line[line.find(start_json):].decode("utf-8"))["token"]["text"]
                        if stop is not None:
                            # Uses same approach as below
                            resp_output = enforce_stop_tokens(resp_output, stop)
                        current_completion += resp_output
                        run_manager.on_llm_new_token(resp_output)
                return current_completion
            except Exception as e:
                raise ValueError(f"Error raised by streaming inference endpoint: {e}")
        else:
            try:
                response = self.client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=content_type,
                    Accept=accepts,
                    **_endpoint_kwargs,
                )
            except Exception as e:
                raise ValueError(f"Error raised by inference endpoint: {e}")

            text = self.content_handler.transform_output(response["Body"])
            if stop is not None:
                # This is a bit hacky, but I can't figure out a better way to enforce
                # stop tokens when making calls to the sagemaker endpoint.
                text = enforce_stop_tokens(text, stop)

            return text
        
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools for use in model generation.
        
        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: If provided, which tool for model to call.
                **This parameter is currently ignored.**
            kwargs: Any additional parameters are passed directly to `self.bind(**kwargs)`.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
