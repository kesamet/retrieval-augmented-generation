from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    project_name="my-llm-app",  # Default is 'default'
    endpoint="http://localhost:6006/v1/traces",
)


def tracing():
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
