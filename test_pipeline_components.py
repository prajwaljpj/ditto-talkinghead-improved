#!/usr/bin/env python3
"""
Simple test - directly measure how fast the actual SDK processes frames.
This avoids import/API issues by using the real pipeline.
"""

import time
import numpy as np
import sys
import torch

print("="*70)
print("PIPELINE COMPONENT SPEED TEST")
print("="*70)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Compute: {torch.cuda.get_device_capability(0)}")

# Initialize the actual SDK (which will load all models)
print("\nInitializing SDK...")
from stream_pipeline_offline import StreamSDK

cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
data_root = "./checkpoints/ditto_trt_custom2"

SDK = StreamSDK(cfg_pkl, data_root)

# Create dummy image source
import cv2
test_img = np.ones((512, 512, 3), dtype=np.uint8) * 128
cv2.imwrite("test_img_speed.jpg", test_img)

print("Setting up pipeline...")
SDK.setup("test_img_speed.jpg", "speed_test.mp4")

print(f"\nOnline mode: {SDK.online_mode}")
print(f"Overlap: {SDK.overlap_v2}")
print(f"Sampling timesteps: {SDK.sampling_timesteps}")

# Test individual components with real data
print("\n" + "="*70)
print("TESTING INDIVIDUAL COMPONENTS")
print("="*70)

# Test 1: Wav2Feat (Hubert)
print("\n1. Wav2Feat (Hubert Audio Encoder)")
print("-" * 70)

try:
    # 10-frame audio chunk
    audio_chunk = np.random.randn(6480).astype(np.float32)

    # Warmup
    for _ in range(5):
        SDK.wav2feat.wav2feat(audio_chunk, sr=16000, chunksize=(3, 5, 2))

    # Measure
    times = []
    for _ in range(20):
        start = time.time()
        feat = SDK.wav2feat.wav2feat(audio_chunk, sr=16000, chunksize=(3, 5, 2))
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"  Average: {avg*1000:.2f} ms per 10-frame chunk")
    print(f"  Per-frame: {avg/10*1000:.2f} ms")
    print(f"  Throughput: {10/avg:.2f} fps")

    if avg > 0.1:
        print("  ⚠️  Hubert is slow!")
    else:
        print("  ✓ Hubert is fast")

except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 2: Audio2Motion (LMDM) - This is the suspected bottleneck
print("\n2. Audio2Motion (LMDM Diffusion Model)")
print("-" * 70)

try:
    # Create proper input using the actual condition handler
    # Generate 80 frames of audio features
    audio_80frames = np.random.randn(51200).astype(np.float32)  # 80 * 640 samples
    aud_feat = SDK.wav2feat.wav2feat(audio_80frames, sr=16000)

    # Pad if needed
    if len(aud_feat) < SDK.audio2motion.seq_frames:
        pad_len = SDK.audio2motion.seq_frames - len(aud_feat)
        aud_feat = np.concatenate([aud_feat, np.zeros((pad_len, aud_feat.shape[1]))], axis=0)
    else:
        aud_feat = aud_feat[:SDK.audio2motion.seq_frames]

    # Get condition
    aud_cond = SDK.condition_handler(aud_feat, 0)[None]

    # Warmup
    print("  Warming up (3 iterations)...")
    for _ in range(3):
        SDK.audio2motion(aud_cond, None)

    # Measure
    print("  Measuring (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        output = SDK.audio2motion(aud_cond, None)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Iteration {i+1}: {elapsed*1000:.0f} ms")

    avg = sum(times) / len(times)
    valid_frames = SDK.audio2motion.valid_clip_len

    print(f"\n  Average: {avg*1000:.0f} ms per inference")
    print(f"  Output frames: {valid_frames} per inference")
    print(f"  Effective throughput: {valid_frames/avg:.2f} fps")

    if avg > 0.5:
        print(f"  ⚠️  LMDM IS VERY SLOW! ({avg*1000:.0f}ms)")
        print(f"  This is your bottleneck!")
        print(f"  To fix: Reduce sampling_timesteps or use FP16")
    elif avg > 0.2:
        print(f"  ⚠️  LMDM is slow ({avg*1000:.0f}ms)")
    else:
        print("  ✓ LMDM is acceptable")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: WarpNetwork
print("\n3. WarpNetwork (3D Warping)")
print("-" * 70)

try:
    # Get real source features
    x_s_info = SDK.source_info['x_s_info_lst'][0]
    f_s = SDK.source_info['f_s_lst'][0]

    # Create dummy driving motion
    x_d = x_s_info['kp'].copy()

    # Warmup
    for _ in range(10):
        SDK.warp_f3d(f_s, x_s_info['kp'], x_d)

    # Measure
    start = time.time()
    for _ in range(100):
        SDK.warp_f3d(f_s, x_s_info['kp'], x_d)
    elapsed = time.time() - start

    avg = elapsed / 100
    print(f"  Average: {avg*1000:.2f} ms per frame")
    print(f"  Throughput: {100/elapsed:.2f} fps")

    if avg > 0.04:
        print("  ⚠️  WarpNetwork is slow!")
    else:
        print("  ✓ WarpNetwork is fast")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Decoder
print("\n4. Decoder (Image Renderer)")
print("-" * 70)

try:
    # Get a real warped feature from previous test
    f_3d = SDK.warp_f3d(f_s, x_s_info['kp'], x_d)

    # Warmup
    for _ in range(10):
        SDK.decode_f3d(f_3d)

    # Measure
    start = time.time()
    for _ in range(100):
        SDK.decode_f3d(f_3d)
    elapsed = time.time() - start

    avg = elapsed / 100
    print(f"  Average: {avg*1000:.2f} ms per frame")
    print(f"  Throughput: {100/elapsed:.2f} fps")

    if avg > 0.04:
        print("  ⚠️  Decoder is slow!")
    else:
        print("  ✓ Decoder is fast")

except Exception as e:
    print(f"  ✗ Failed: {e}")

# Summary
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print("""
For RTX 4090, expected speeds:
  Hubert:       >200 fps (very fast)
  LMDM:         5-20 fps (bottleneck, depends on sampling_timesteps)
  WarpNetwork:  >50 fps (fast)
  Decoder:      >80 fps (fast)

Your config:
  sampling_timesteps: 10 (high - slower but better quality)
  overlap_v2: 70 (very high - much slower)

To speed up:
  1. Reduce sampling_timesteps: 10 → 5 (2x faster LMDM)
  2. Reduce overlap_v2: 70 → 10 (uses smaller context)
  3. Rebuild LMDM in FP16 (2x faster)
  4. Remove Ampere_Plus compatibility mode (10-20% faster)
""")

# Cleanup
SDK.stop_event.set()
print("\n✓ Test complete!")
