import logging

import box
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    logger.info("Reading config.yaml")
    with open("config.yaml", "r") as f:
        CFG = box.Box(yaml.safe_load(f))

except Exception as e:
    logger.error(e)


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
