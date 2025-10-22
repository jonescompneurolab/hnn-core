import os
import json

CONFIG_FILE = os.path.expanduser("~/.hnn_core_config.json")

def check_first_run():
    if not os.path.exists(CONFIG_FILE):
        print("Thank you for installing HNN-Core!")
        print("Please fill out our survey at:")
        print(
            "https://docs.google.com/forms/d/e/1FAIpQLSfN2F4IkGATs6cy1QBO78C6QJqvm9y14TqsCUsuR4Rrkmr1Mg/viewform"
        )
        with open(CONFIG_FILE, "w") as f:
            json.dump({"survey_seen": True}, f)
