import box
import yaml
from dotenv import load_dotenv

with open("config.yaml", "r") as f:
    CFG = box.Box(yaml.safe_load(f))

_ = load_dotenv()
