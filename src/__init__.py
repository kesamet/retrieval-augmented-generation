import box
import yaml

with open("config.yaml", "r", encoding="utf8") as f:
    CFG = box.Box(yaml.safe_load(f))
