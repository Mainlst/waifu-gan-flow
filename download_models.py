import os
import requests
from tqdm import tqdm

MODEL_DIR = "models"
FILES = {
    "g_mapping.onnx": "https://huggingface.co/skytnt/waifu-gan/resolve/main/g_mapping.onnx",
    "g_synthesis.onnx": "https://huggingface.co/skytnt/waifu-gan/resolve/main/g_synthesis.onnx"
}

os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"[✓] {filepath} already exists. Skipping.")
        return
    print(f"Downloading {filepath}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
    print(f"[✓] Downloaded {filepath}")

if __name__ == "__main__":
    for filename, url in FILES.items():
        download_file(url, os.path.join(MODEL_DIR, filename))
