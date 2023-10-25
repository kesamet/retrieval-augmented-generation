import box
import torch
import yaml

with open("config.yaml", "r") as f:
    CFG = box.Box(yaml.safe_load(f))

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    CFG.DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    CFG.DEVICE = torch.device("cuda")
else:
    CFG.DEVICE = torch.device("cpu")
