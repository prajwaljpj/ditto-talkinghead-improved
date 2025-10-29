#!/usr/bin/env python3
"""
Quick test to identify which TensorRT engine is causing the hang.
This loads each engine one-by-one with verbose output.
"""

import sys
import time
import tensorrt as trt
from pathlib import Path

def test_single_engine(engine_path, timeout=60):
    """Test loading a single TensorRT engine with timeout."""
    print(f"\n{'='*70}")
    print(f"Testing: {engine_path.name}")
    print(f"{'='*70}")

    if not engine_path.exists():
        print(f"  ✗ File not found")
        return False

    file_size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {file_size_mb:.2f} MB")

    try:
        print(f"  Loading engine... ", end="", flush=True)
        start_time = time.time()

        # Create logger
        logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine_data = f.read()

            # This is where it often hangs
            engine = runtime.deserialize_cuda_engine(engine_data)

        elapsed = time.time() - start_time

        if engine:
            print(f"✓ ({elapsed:.2f}s)")

            # Handle both TensorRT 8.x (num_bindings) and 10.x (num_io_tensors) APIs
            try:
                if hasattr(engine, 'num_io_tensors'):
                    # TensorRT 10.x API
                    num_io = engine.num_io_tensors
                    print(f"  I/O tensors: {num_io}")
                elif hasattr(engine, 'num_bindings'):
                    # TensorRT 8.x API
                    print(f"  Inputs: {engine.num_bindings // 2}")
                    print(f"  Outputs: {engine.num_bindings // 2}")
            except:
                pass  # Just skip the details

            # Free engine
            del engine
            return True
        else:
            print(f"✗ Failed to deserialize")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ({elapsed:.2f}s)")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test TensorRT engine loading")
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_trt_custom",
                       help="Path to TensorRT engines directory")

    args = parser.parse_args()
    data_root = Path(args.data_root)

    print("\n" + "="*70)
    print("  TENSORRT ENGINE LOADING TEST")
    print("="*70)
    print(f"Directory: {data_root}")

    if not data_root.exists():
        print(f"\n✗ Directory not found: {data_root}")
        return 1

    # List of engines to test in order of importance/likelihood to hang
    engines_to_test = [
        "hubert_fp32.engine",              # Most likely to hang
        "lmdm_v0.4_hubert_fp32.engine",    # Second most likely
        "motion_extractor_fp32.engine",
        "appearance_extractor_fp16.engine",
        "warp_network_fp16.engine",
        "decoder_fp16.engine",
        "stitch_network_fp16.engine",
        "face_mesh_fp16.engine",
        "blaze_face_fp16.engine",
        "insightface_det_fp16.engine",
        "landmark106_fp16.engine",
        "landmark203_fp16.engine",
    ]

    results = {}

    for engine_name in engines_to_test:
        engine_path = data_root / engine_name
        if engine_path.exists():
            results[engine_name] = test_single_engine(engine_path)
        else:
            print(f"\n⊘ Skipping {engine_name} (not found)")

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    passed = [name for name, result in results.items() if result]
    failed = [name for name, result in results.items() if not result]

    print(f"\n  ✓ Passed: {len(passed)}/{len(results)}")
    print(f"  ✗ Failed: {len(failed)}/{len(results)}")

    if failed:
        print("\n  Failed engines:")
        for name in failed:
            print(f"    - {name}")
        print("\n  ⚠ These engines need to be rebuilt!")
        print("="*70)
        return 1
    else:
        print("\n  ✓ All engines loaded successfully!")
        print("="*70)
        return 0

if __name__ == "__main__":
    sys.exit(main())
