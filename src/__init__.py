import box
import yaml

with open("config.yaml", "r") as f:
    CFG = box.Box(yaml.safe_load(f))

try:
    from dotenv import load_dotenv

    _ = load_dotenv()
except ModuleNotFoundError:
    pass
