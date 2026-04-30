#!/usr/bin/env python3
"""
Face Swap Batch - Main entry point.

Usage:
    # Interactive (guided setup)
    python main.py --setup

    # With saved configuration (press Enter to use last values)
    python main.py

    # Local (CPU fallback)
    python main.py --source foto.jpg --input input/ --output output/

    # Con Replicate API (recomendado para resultados reales)
    python main.py --source foto.jpg --input input/ --output output/ --api-token TU_TOKEN
"""

import argparse
import os
import sys
from pathlib import Path

from src import FaceDetector, process_batch
from src.face_swap import FaceSwapper
from src.replicate_swap import process_batch_replicate
from src.config import load_config, interactive_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Swap Batch - Swap faces in images by batch"
    )
    parser.add_argument(
        "--setup", "-S",
        action="store_true",
        help="Run interactive configuration wizard"
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        default=None,
        help="Source image path (face to swap from)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Input directory with target images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Batch size (default: 10)"
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="Replicate API token (for cloud processing)"
    )
    parser.add_argument(
        "--replicate-model", "-r",
        type=str,
        default=None,
        help="Replicate model identifier (e.g., ddvinh1/inswapper:hash)"
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=None,
        help="Path to face swapper ONNX model (optional, local)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for local inference if available"
    )

    return parser.parse_args()


def get_providers(use_gpu: bool):
    """Get ONNX providers based on availability."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        if use_gpu and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    except ImportError:
        return ["CPUExecutionProvider"]


def main():
    args = parse_args()

    # Interactive setup mode
    if args.setup:
        config = interactive_config()
        print("\nConfiguration saved!")
        return

    # Load saved config
    config = load_config()

    # Resolve paths - use CLI args or fall back to saved config
    source_path = args.source or (Path(config["source"]) if config.get("source") else None)
    input_dir = args.input or (Path(config["input_dir"]) if config.get("input_dir") else None)
    output_dir = args.output or (Path(config["output_dir"]) if config.get("output_dir") else None)

    # Validate paths
    if not source_path or not source_path.exists():
        print(f"Error: Source image not found: {source_path}")
        print("Run with --setup to configure, or provide --source")
        sys.exit(1)

    if not input_dir or not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        print("Run with --setup to configure, or provide --input")
        sys.exit(1)

    # Check for API token (优先idad a CLI > config > env)
    api_token = args.api_token or config.get("api_token") or os.environ.get("REPLICATE_API_TOKEN")

    # Get model from config or CLI override
    model_name = args.replicate_model or config.get("model", "ddvinh1/inswapper:25bdae46f2713138640b6e8c04dc4ca18625ce95b1863936b053eee42d9ba6db")

    if api_token:
        # Use Replicate API
        print(f"\n=== Using Replicate API (Cloud) ===")
        print(f"Model: {model_name}")
        print(f"  Source: {source_path}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")

        stats = process_batch_replicate(
            source_path=str(source_path),
            input_dir=input_dir,
            output_dir=output_dir,
            api_token=api_token,
            model=model_name,
            batch_size=args.batch_size
        )

        if stats["processed"] > 0:
            print(f"\n✓ Done! {stats['processed']} images processed")
            print(f"  Estimated cost: ${stats['cost']:.4f}")
            sys.exit(0)
        else:
            print(f"\n✗ No images processed")
            sys.exit(1)

    else:
        # Use local processing
        print(f"\n=== Using Local Processing (CPU) ===")

        providers = get_providers(args.use_gpu)
        print(f"Providers: {providers}")

        # Look for model
        model_dir = Path(__file__).parent / "models"
        default_model = model_dir / "inswapper_128.onnx"
        model_path = args.model or default_model

        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            print("Using fallback (no actual face swap)...")
            print("Note: Pass --api-token for real face swap via Replicate")

        swapper = FaceSwapper(
            model_path=model_path,
            providers=providers
        )

        print("Initializing face detector (OpenCV fallback)...")
        detector = FaceDetector(providers=providers)

        print(f"\nProcessing batch...")

        stats = process_batch(
            source_path=source_path,
            input_dir=input_dir,
            output_dir=output_dir,
            detector=detector,
            swapper=swapper,
            batch_size=args.batch_size
        )

        if stats["processed"] > 0:
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()