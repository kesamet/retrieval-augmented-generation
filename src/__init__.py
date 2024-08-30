from loguru import logger
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

logger.info("Reading config.yaml")
CFG = OmegaConf.load("config.yaml")

if CFG.USE_TRACING:
    from src.tracing import tracing

    tracing()
