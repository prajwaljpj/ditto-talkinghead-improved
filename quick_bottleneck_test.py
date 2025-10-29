#!/usr/bin/env python3
"""
Quick test to identify which component is slow.
Tests each component in isolation.
"""

import time
import numpy as np
import sys

def test_audio2motion():
    """Test LMDM speed."""
    print("\n" + "="*70)
    print("Testing audio2motion (LMDM)")
    print("="*70)

    from core.models.lmdm import LMDM

    cfg = {
        'model_path': './checkpoints/ditto_trt_custom/lmdm_v0.4_hubert_fp32.engine',
        'device': 'cuda',
        'motion_feat_dim': 265,
        'audio_feat_dim': 1024 + 35,
        'seq_frames': 80,
        'sampling_timesteps': 10,  # From your config
    }

    print("Initializing LMDM...")
    lmdm = LMDM(**cfg)

    # Dummy input
    audio_feat = np.random.randn(1, 80, 1059).astype(np.float32)

    # Warmup
    print("Warmup (3 iterations)...")
    for _ in range(3):
        lmdm(audio_feat, None)

    # Measure
    print("Measuring (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        output = lmdm(audio_feat, None)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.0f} ms")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg*1000:.0f} ms per inference")
    print(f"Throughput: {1.0/avg:.2f} inferences/s")
    print(f"Output frames per inference: ~10 (valid_clip_len)")
    print(f"Effective frame rate: {10.0/avg:.2f} fps")

    if avg > 0.5:
        print("⚠️  SLOW! This is your bottleneck!")
        print("   Recommendation: Reduce sampling_timesteps from 10 to 5")


def test_warp_network():
    """Test warp network speed."""
    print("\n" + "="*70)
    print("Testing warp_f3d (WarpNetwork)")
    print("="*70)

    from core.models.warp_trt import WarpNetworkTRT

    cfg = {
        'model_path': './checkpoints/ditto_trt_custom/warp_network_fp16.engine',
        'device': 'cuda'
    }

    print("Initializing WarpNetwork...")
    warp = WarpNetworkTRT(**cfg)

    # Dummy input
    f_s = np.random.randn(1, 32, 64, 64).astype(np.float32)
    x_s = np.random.randn(1, 63, 3).astype(np.float32)
    x_d = np.random.randn(1, 63, 3).astype(np.float32)

    # Warmup
    print("Warmup (10 iterations)...")
    for _ in range(10):
        warp(f_s, x_s, x_d)

    # Measure
    print("Measuring (100 iterations)...")
    start = time.time()
    for _ in range(100):
        warp(f_s, x_s, x_d)
    elapsed = time.time() - start

    avg = elapsed / 100
    print(f"\nAverage: {avg*1000:.2f} ms per frame")
    print(f"Throughput: {100/elapsed:.2f} fps")

    if avg > 0.04:  # 40ms = 25fps
        print("⚠️  SLOW! This is a bottleneck!")


def test_decoder():
    """Test decoder speed."""
    print("\n" + "="*70)
    print("Testing decode_f3d (Decoder)")
    print("="*70)

    from core.models.decoder_trt import DecoderTRT

    cfg = {
        'model_path': './checkpoints/ditto_trt_custom/decoder_fp16.engine',
        'device': 'cuda'
    }

    print("Initializing Decoder...")
    decoder = DecoderTRT(**cfg)

    # Dummy input
    f_3d = np.random.randn(1, 32, 64, 64).astype(np.float32)

    # Warmup
    print("Warmup (10 iterations)...")
    for _ in range(10):
        decoder(f_3d)

    # Measure
    print("Measuring (100 iterations)...")
    start = time.time()
    for _ in range(100):
        decoder(f_3d)
    elapsed = time.time() - start

    avg = elapsed / 100
    print(f"\nAverage: {avg*1000:.2f} ms per frame")
    print(f"Throughput: {100/elapsed:.2f} fps")

    if avg > 0.04:
        print("⚠️  SLOW! This is a bottleneck!")


def test_hubert():
    """Test Hubert audio encoder speed."""
    print("\n" + "="*70)
    print("Testing Hubert (Audio Encoder)")
    print("="*70)

    from core.models.hubert_trt import HubertTRT

    cfg = {
        'model_path': './checkpoints/ditto_trt_custom/hubert_fp32.engine',
        'device': 'cuda',
    }

    print("Initializing Hubert...")
    hubert = HubertTRT(**cfg)

    # Dummy input (10 frames of audio = 6480 samples)
    audio = np.random.randn(6480).astype(np.float32)

    # Warmup
    print("Warmup (5 iterations)...")
    for _ in range(5):
        hubert(audio)

    # Measure
    print("Measuring (20 iterations)...")
    times = []
    for i in range(20):
        start = time.time()
        hubert(audio)
        elapsed = time.time() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg*1000:.0f} ms per 10-frame chunk")
    print(f"Per-frame: {avg/10*1000:.2f} ms")
    print(f"Throughput: {10/avg:.2f} fps")

    if avg > 0.1:
        print("⚠️  SLOW! Hubert is a bottleneck!")


def main():
    print("="*70)
    print("COMPONENT BOTTLENECK TEST")
    print("="*70)
    print("\nThis will test each pipeline component individually.")
    print("This helps identify which specific model is slow.\n")

    try:
        test_hubert()
    except Exception as e:
        print(f"✗ Hubert test failed: {e}")

    try:
        test_audio2motion()
    except Exception as e:
        print(f"✗ Audio2motion test failed: {e}")

    try:
        test_warp_network()
    except Exception as e:
        print(f"✗ Warp network test failed: {e}")

    try:
        test_decoder()
    except Exception as e:
        print(f"✗ Decoder test failed: {e}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nCheck the output above for ⚠️  SLOW! warnings.")
    print("The slowest component is your primary bottleneck.")
    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
