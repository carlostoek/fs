#!/usr/bin/env python3
"""
Face Swap Batch - Main entry point.

Usage:
    # Local (CPU fallback - sin face swap real)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Swap Batch - Swap faces in images by batch"
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        required=True,
        help="Source image path (face to swap from)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory with target images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
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

    # Validate paths
    if not args.source.exists():
        print(f"Error: Source image not found: {args.source}")
        sys.exit(1)

    if not args.input.exists() or not args.input.is_dir():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    # Check for API token (优先idad a Replicate)
    api_token = args.api_token or os.environ.get("REPLICATE_API_TOKEN")

    if api_token:
        # Use Replicate API
        print(f"\n=== Using Replicate API (Cloud) ===")
        print(f"Model: codeplugtech/face-swap")
        print(f"  Source: {args.source}")
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output}")

        stats = process_batch_replicate(
            source_path=str(args.source),
            input_dir=args.input,
            output_dir=args.output,
            api_token=api_token,
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
            source_path=args.source,
            input_dir=args.input,
            output_dir=args.output,
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