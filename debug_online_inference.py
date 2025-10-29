#!/usr/bin/env python3
"""
Debug script for online inference issues.
Helps identify where the inference is getting stuck.
"""

import sys
import time
import threading
import traceback
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_imports():
    """Test that all required modules can be imported."""
    print_section("1. Testing Module Imports")

    modules = [
        ("torch", "PyTorch"),
        ("tensorrt", "TensorRT"),
        ("librosa", "Librosa"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False

    return all_ok

def test_cuda():
    """Test CUDA availability."""
    print_section("2. Testing CUDA")

    try:
        import torch
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        return torch.cuda.is_available()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_tensorrt_engines(data_root):
    """Test loading TensorRT engines with timeout."""
    print_section("3. Testing TensorRT Engine Loading")

    engine_files = [
        "hubert_fp32.engine",  # Often the slowest
        "lmdm_v0.4_hubert_fp32.engine",
        "appearance_extractor_fp16.engine",
        "motion_extractor_fp32.engine",
        "warp_network_fp16.engine",
        "decoder_fp16.engine",
    ]

    import tensorrt as trt

    for engine_file in engine_files:
        engine_path = Path(data_root) / engine_file
        if not engine_path.exists():
            print(f"  ⊘ {engine_file}: Not found (skipping)")
            continue

        print(f"  Testing {engine_file}... ", end="", flush=True)

        # Test with timeout
        success = [False]
        error_msg = [None]

        def load_engine():
            try:
                with open(engine_path, "rb") as f:
                    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                    engine = runtime.deserialize_cuda_engine(f.read())
                    if engine:
                        success[0] = True
                    else:
                        error_msg[0] = "Failed to deserialize"
            except Exception as e:
                error_msg[0] = str(e)

        thread = threading.Thread(target=load_engine)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # 30 second timeout

        if thread.is_alive():
            print("✗ TIMEOUT (>30s) - This is likely where it's hanging!")
            return False
        elif success[0]:
            print("✓")
        else:
            print(f"✗ Error: {error_msg[0]}")
            return False

    return True

def test_config(cfg_pkl):
    """Test config file loading."""
    print_section("4. Testing Config File")

    if not Path(cfg_pkl).exists():
        print(f"  ✗ Config file not found: {cfg_pkl}")
        return False

    try:
        import pickle
        with open(cfg_pkl, "rb") as f:
            cfg = pickle.load(f)

        print(f"  ✓ Config loaded successfully")

        # Check for online mode flag
        if "online_mode" in cfg.get("default_kwargs", {}):
            online_mode = cfg["default_kwargs"]["online_mode"]
            print(f"  Online mode: {online_mode}")
            if not online_mode:
                print("  ⚠ WARNING: online_mode=False in config!")
                print("  You should use v0.4_hubert_cfg_trt_online.pkl")
        else:
            print("  ⚠ WARNING: online_mode not found in config")

        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        traceback.print_exc()
        return False

def test_sdk_init(cfg_pkl, data_root):
    """Test SDK initialization with timeout."""
    print_section("5. Testing SDK Initialization")

    success = [False]
    error_msg = [None]

    def init_sdk():
        try:
            from stream_pipeline_offline import StreamSDK
            print("  Loading SDK... ", end="", flush=True)
            sdk = StreamSDK(cfg_pkl, data_root)
            success[0] = True
            print("✓")
        except Exception as e:
            error_msg[0] = str(e)
            traceback.print_exc()

    thread = threading.Thread(target=init_sdk)
    thread.daemon = True
    thread.start()

    # Wait with progress indicator
    for i in range(60):  # 60 second timeout
        if not thread.is_alive():
            break
        if i % 5 == 0:
            print(f"  ... {i}s", flush=True)
        time.sleep(1)

    if thread.is_alive():
        print("  ✗ TIMEOUT (>60s) - SDK initialization hung!")
        print("  This usually means a TensorRT engine is hanging during load")
        return False
    elif success[0]:
        print("  ✓ SDK initialized successfully")
        return True
    else:
        print(f"  ✗ Error: {error_msg[0]}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug online inference hanging issues")
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_trt_custom",
                       help="Path to model directory")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                       help="Path to config file")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ONLINE INFERENCE DEBUG TOOL")
    print("=" * 70)
    print(f"Data Root: {args.data_root}")
    print(f"Config: {args.cfg_pkl}")

    # Run tests
    results = []

    results.append(("Module Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Config File", test_config(args.cfg_pkl)))
    results.append(("TensorRT Engines", test_tensorrt_engines(args.data_root)))
    results.append(("SDK Initialization", test_sdk_init(args.cfg_pkl, args.data_root)))

    # Summary
    print_section("SUMMARY")

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("  ✓ All tests passed!")
        print("  Your setup should work for online inference.")
    else:
        print("  ✗ Some tests failed!")
        print("  See above for details on what needs to be fixed.")

    print("=" * 70 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
