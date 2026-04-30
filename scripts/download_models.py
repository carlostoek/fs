#!/usr/bin/env python3
"""
Download models for face swap.

Note: The inswapper model has licensing restrictions and is not openly available.
This script provides paths to download the required models from legitimate sources.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# Model sources - these are publicly available face analysis models
MODELS = {
    "retinaface": {
        "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/retinaface/retinaface_r50_v1.onnx",
        "filename": "retinaface_r50_v1.onnx",
        "description": "Face detector - RetinaFace R50",
    },
    # inswapper - not publicly available due to licensing
    # Obtain from: facefusion, comfyui, or reactor (legitimate sources)
}

def download_file(url: str, dest: Path) -> bool:
    """Download a file using curl."""
    import subprocess

    print(f"Downloading {dest.name}...")
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        return False
    except FileNotFoundError:
        print("curl not found. Please install curl or download manually.")
        return False

def main():
    print("Face Swap Model Downloader")
    print("=" * 40)
    print(f"Models directory: {MODELS_DIR}")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Available models for download:")
    for name, info in MODELS.items():
        print(f"  - {name}: {info['description']}")
        print(f"    -> {info['url']}")

    print()
    print("Models requiring manual download (licensing restrictions):")
    print("  - inswapper_128.onnx: Get from facefusion/comfyui/reactor")
    print()

    # Download available models
    for name, info in MODELS.items():
        dest = MODELS_DIR / info["filename"]
        if dest.exists():
            print(f"{name}: Already exists")
            continue

        if download_file(info["url"], dest):
            print(f"{name}: Downloaded successfully")
        else:
            print(f"{name}: Download failed")

if __name__ == "__main__":
    main()