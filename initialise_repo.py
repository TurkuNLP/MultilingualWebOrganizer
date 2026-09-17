# Initialisation script that will create folders for the template.
# Use with --delete flag to also delete this script and clear README.md for a clean start.
import os
from sys import argv

folders = (
    "data",
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "environments",
    "environments/lumi",
    "environments/roihu",
    "hpc",
    "hpc/lumi",
    "hpc/roihu",
    "models",
    "configs",
    "scripts",
    "scripts/data_processing",
    "scripts/data_analysis",
    "scripts/model_training",
    "scripts/model_evaluation",
    "results",
    "results/figures",
    "cache",
    "logs",
)

if __name__ == "__main__":
    print("Creating folders...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Folders created successfully.")
    if len(argv) > 1 and argv[1] == "--delete":
        print("Clearing README.md")
        with open("README.md", "w") as f:
            f.write("")
        print("Deleting the initialisation script...")
        os.remove(__file__)
