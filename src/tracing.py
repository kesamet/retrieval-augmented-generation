import requests
from loguru import logger


def tracing():
    url = "http://localhost:6006"
    try:
        _ = requests.get(url)
        logger.info("Phoenix UI: http://localhost:6006\nLog traces: /v1/traces over HTTP")
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        logger.error("Phoenix server not started. Skipped tracing.")
    else:
        from phoenix.trace.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
