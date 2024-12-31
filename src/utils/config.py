import json

with open("./src/config/config.json", "r") as f:
    config : dict = json.load(f)