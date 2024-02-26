import logging

import box
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    CFG = box.Box(yaml.safe_load(f))
