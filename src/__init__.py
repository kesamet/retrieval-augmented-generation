import requests
from loguru import logger
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

logger.info("Reading config.yaml")
CFG = OmegaConf.load("config.yaml")

if CFG.USE_TRACING:
    # Setup tracing
    url = "http://localhost:6006"
    try:
        page = requests.get(url)
        logger.info(
            "Phoenix UI: http://localhost:6006\nLog traces: /v1/traces over HTTP"
        )
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        logger.error("Phoenix server not started. Skipped tracing.")
    else:
        from phoenix.trace.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument()
