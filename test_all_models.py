#!/usr/bin/env python3
"""Test speed of all TensorRT models individually."""

import time
import numpy as np
import torch

print("="*70)
print("INDIVIDUAL MODEL SPEED TEST")
print("="*70)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Compute: {torch.cuda.get_device_capability(0)}")
print(f"TensorRT: {torch.version.cuda}")

# Test 1: Hubert (Audio Encoder)
print("\n" + "="*70)
print("1. Hubert (Audio Encoder)")
print("="*70)

try:
    from core.aux_models.hubert_stream import HubertStreaming

    cfg = {'model_path': './checkpoints/ditto_trt_custom2/hubert_fp32.engine', 'device': 'cuda'}
    hubert = HubertStreaming(**cfg)

    # Test with 10-frame audio chunk (6480 samples)
    audio = np.random.randn(6480).astype(np.float32)

    # Warmup
    for _ in range(5):
        hubert(audio)

    # Measure
    times = []
    for _ in range(20):
        start = time.time()
        hubert(audio)
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"Average: {avg*1000:.2f} ms per 10-frame chunk")
    print(f"Per-frame: {avg/10*1000:.2f} ms")
    print(f"Throughput: {10/avg:.2f} fps")

    if avg > 0.1:
        print("⚠️  Hubert is slow!")
    else:
        print("✓ Hubert is fast")

except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: LMDM (Audio2Motion)
print("\n" + "="*70)
print("2. LMDM (Audio2Motion - Diffusion Model)")
print("="*70)

try:
    from core.models.lmdm import LMDM

    cfg = {
        'model_path': './checkpoints/ditto_trt_custom2/lmdm_v0.4_hubert_fp32.engine',
        'device': 'cuda',
        'motion_feat_dim': 265,
        'audio_feat_dim': 1024 + 35,
        'seq_frames': 80,
    }

    lmdm = LMDM(**cfg)
    lmdm.setup(sampling_timesteps=10)  # Your current config

    # Dummy inputs
    kp_cond = np.random.randn(1, 265).astype(np.float32)  # Initial keypoint condition
    audio_cond = np.random.randn(1, 80, 1059).astype(np.float32)  # 80 frames of audio features

    # Warmup
    print("Warming up (3 iterations)...")
    for _ in range(3):
        lmdm(kp_cond, audio_cond, sampling_timesteps=10)

    # Measure
    print("Measuring (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        output = lmdm(kp_cond, audio_cond, sampling_timesteps=10)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.0f} ms")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg*1000:.0f} ms per inference")
    print(f"Output: ~10 valid frames per inference")
    print(f"Effective: {10/avg:.2f} fps")

    if avg > 0.5:
        print("⚠️  LMDM is SLOW! This is your bottleneck!")
    else:
        print("✓ LMDM is acceptable")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: WarpNetwork
print("\n" + "="*70)
print("3. WarpNetwork (3D Warping)")
print("="*70)

try:
    from core.models.warp_network import WarpNetwork

    cfg = {'model_path': './checkpoints/ditto_trt_custom2/warp_network_fp16.engine', 'device': 'cuda'}
    warp = WarpNetwork(**cfg)

    # Dummy input - correct shapes based on the model
    feature_3d = np.random.randn(1, 32, 16, 64, 64).astype(np.float32)  # 3D feature volume
    kp_source = np.random.randn(1, 21, 3).astype(np.float32)  # 21 keypoints (not 63)
    kp_driving = np.random.randn(1, 21, 3).astype(np.float32)

    # Warmup
    for _ in range(10):
        warp(feature_3d, kp_source, kp_driving)

    # Measure
    start = time.time()
    for _ in range(100):
        warp(feature_3d, kp_source, kp_driving)
    elapsed = time.time() - start

    avg = elapsed / 100
    print(f"Average: {avg*1000:.2f} ms per frame")
    print(f"Throughput: {100/elapsed:.2f} fps")

    if avg > 0.04:
        print("⚠️  WarpNetwork is slow!")
    else:
        print("✓ WarpNetwork is fast")

except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: Decoder (already tested)
print("\n" + "="*70)
print("4. Decoder (Image Renderer)")
print("="*70)

try:
    from core.models.decoder import Decoder

    cfg = {'model_path': './checkpoints/ditto_trt_custom2/decoder_fp16.engine', 'device': 'cuda'}
    decoder = Decoder(**cfg)

    # Dummy input
    f_3d = np.random.randn(1, 256, 64, 64).astype(np.float32)

    # Warmup
    for _ in range(10):
        decoder(f_3d)

    # Measure
    start = time.time()
    for _ in range(100):
        decoder(f_3d)
    elapsed = time.time() - start

    avg = elapsed / 100
    print(f"Average: {avg*1000:.2f} ms per frame")
    print(f"Throughput: {100/elapsed:.2f} fps")

    if avg > 0.04:
        print("⚠️  Decoder is slow!")
    else:
        print("✓ Decoder is fast")

except Exception as e:
    print(f"✗ Failed: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Expected results for RTX 4090:
  Hubert:       <50ms per 10 frames (fast)
  LMDM:         200-500ms per inference (expected bottleneck)
  WarpNetwork:  <20ms per frame (fast)
  Decoder:      <15ms per frame (fast)

If LMDM takes >500ms, it's the bottleneck.
If all models are fast but pipeline is slow, it's a pipeline issue.
""")
