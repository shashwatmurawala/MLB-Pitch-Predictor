import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("janus137/mlb-postseason-2025-pitch-by-pitch-data")

print("Path to dataset files:", path)

# Copy the file to the current directory for easier access
target_dir = "./data"
os.makedirs(target_dir, exist_ok=True)

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)

print("Data successfully copied to ./data/")
