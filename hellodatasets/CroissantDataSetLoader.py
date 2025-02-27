import json
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import os

# Path to the Croissant manifest file
manifest_path = "../firstcroissant/croissant_manifest.json"

# Load the manifest
with open(manifest_path, "r") as f:
    manifest = json.load(f)

# Function to load each split from a JSONL file
def load_split(file_path):
    # We use the "json" loading script; each line in file_path is a JSON record.
    return load_dataset("json", data_files=file_path, split="train")

# Load each split based on the manifest
train_dataset = load_split(manifest["splits"]["train"])
val_dataset = load_split(manifest["splits"]["validation"])
test_dataset = load_split(manifest["splits"]["test"])

# Combine splits into a DatasetDict
croissant_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Inspect the dataset
print(croissant_dataset)

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN_CREATE")
login(token=os.environ["HF_TOKEN"])
croissant_dataset.push_to_hub("arunvelusamyd/hello-dataset")

#Customizations is below
def preprocess(example):
    # Convert label to an integer if needed
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    example["label"] = label_map.get(example["label"], -1)
    return example

croissant_dataset = croissant_dataset.map(preprocess)
#Enable the below line to push the customization
#croissant_dataset.push_to_hub("arunvelusamyd/hello-dataset")