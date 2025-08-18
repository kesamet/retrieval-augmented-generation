import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

px.launch_app()  # TODO

PROJECT_NAME = "my-llm-app"
tracer_provider = register(
    project_name=PROJECT_NAME,  # Default is 'default'
    endpoint="http://localhost:6006/v1/traces",
)


def tracing():
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


def save_traces(filepath):
    """
    Save traces to a file or upload to a bucket.
    If bucket_name is provided, the traces will be uploaded to the specified bucket.
    """
    df = px.Client().get_spans_dataframe(project_name=PROJECT_NAME)
    df.to_csv(filepath, index=False)
