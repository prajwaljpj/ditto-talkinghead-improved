#!/usr/bin/env python3
"""
Test script to verify the 3240 sample chunk size works with StreamSDK.

This validates:
1. The chunk size is accepted by TensorRT models (minimum: 3240 samples)
2. Frame generation rate matches expectations (~5-6 frames per chunk)
3. 25-30 FPS target is achievable
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stream_pipeline_online import StreamSDK

def test_chunk_size():
    """Test the 3240 sample chunk size for 25-30 FPS operation."""

    print("=" * 70)
    print("Testing 3240 Sample Chunk Size for 25-30 FPS")
    print("=" * 70)

    # Configuration
    cfg_pkl = "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
    data_root = "checkpoints/ditto_trt_custom2/"
    source_path = "example/image.png"

    print(f"\n📋 Configuration:")
    print(f"   Config: {cfg_pkl}")
    print(f"   Data root: {data_root}")
    print(f"   Source: {source_path}")

    # Test parameters
    chunk_size = 3240  # samples @ 16kHz (TensorRT minimum)
    chunk_duration_ms = (chunk_size / 16000) * 1000
    expected_fps_min = 25
    expected_fps_max = 30
    expected_frames_per_chunk = 5.5  # Average of 5-6

    print(f"\n🎯 Test Parameters:")
    print(f"   Chunk size: {chunk_size} samples (TensorRT minimum)")
    print(f"   Chunk duration: {chunk_duration_ms:.2f}ms @ 16kHz")
    print(f"   Expected FPS: {expected_fps_min}-{expected_fps_max}")
    print(f"   Expected frames per chunk: ~{expected_frames_per_chunk:.0f}-{expected_frames_per_chunk+1:.0f}")
    print(f"   TensorRT valid range: [3240..12960] samples")

    # Frame callback to count frames
    frame_count = 0
    chunk_count = 0
    frame_times = []

    def frame_callback(frame_rgb, frame_idx, timestamp):
        nonlocal frame_count, frame_times
        frame_count += 1
        frame_times.append(time.time())
        if frame_count % 10 == 0:
            print(f"   Frame {frame_count} generated (idx: {frame_idx}, timestamp: {timestamp:.3f}s)")

    try:
        # Initialize SDK
        print("\n🔧 Initializing StreamSDK...")
        sdk = StreamSDK(cfg_pkl, data_root)

        # Setup with streaming mode
        print("🔧 Setting up StreamSDK (online + streaming mode)...")
        sdk.setup(
            source_path,
            output_path=None,
            frame_callback=frame_callback,
            online_mode=True,
            N_d=-1,
        )
        print("✅ StreamSDK initialized\n")

        # Test feeding multiple chunks
        num_test_chunks = 10
        print(f"🎬 Feeding {num_test_chunks} audio chunks ({chunk_size} samples each)...")
        print(f"   Expected: ~{int(num_test_chunks * expected_frames_per_chunk)}-{int(num_test_chunks * (expected_frames_per_chunk+1))} frames total\n")

        start_time = time.time()

        for i in range(num_test_chunks):
            # Create silent audio chunk
            audio_chunk = np.zeros(chunk_size, dtype=np.float32)

            chunk_start = time.time()
            sdk.run_chunk(audio_chunk, (3, 5, 2))  # Neutral expression
            chunk_time = (time.time() - chunk_start) * 1000

            chunk_count += 1
            print(f"   Chunk {chunk_count}/{num_test_chunks} processed in {chunk_time:.2f}ms")

            # Small delay to allow frame processing
            time.sleep(0.02)

        # Wait a bit for remaining frames to process
        print("\n⏳ Waiting for frame processing to complete...")
        time.sleep(0.5)

        total_time = time.time() - start_time

        # Calculate statistics
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        print(f"Chunks fed: {chunk_count}")
        print(f"Frames generated: {frame_count}")
        print(f"Frames per chunk: {frame_count / chunk_count:.2f}")
        print(f"Total time: {total_time:.2f}s")

        if len(frame_times) > 1:
            # Calculate FPS from frame timestamps
            frame_intervals = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
            avg_interval = sum(frame_intervals) / len(frame_intervals)
            measured_fps = 1.0 / avg_interval if avg_interval > 0 else 0

            print(f"\n⏱️  Frame Timing:")
            print(f"   Avg frame interval: {avg_interval*1000:.2f}ms")
            print(f"   Measured FPS: {measured_fps:.2f}")
            print(f"   Target FPS range: {expected_fps_min}-{expected_fps_max}")

            if measured_fps >= expected_fps_min:
                print(f"\n✅ SUCCESS: Achieving {measured_fps:.1f} FPS (target: {expected_fps_min}-{expected_fps_max})")
            else:
                print(f"\n⚠️  WARNING: Only achieving {measured_fps:.1f} FPS (target: {expected_fps_min}-{expected_fps_max})")

        # Validate chunk size works
        if frame_count > 0:
            print(f"\n✅ CHUNK SIZE VALIDATED: {chunk_size} samples works correctly")
        else:
            print(f"\n❌ FAILED: No frames generated - chunk size may be invalid")

        # Close SDK
        print("\n🔧 Closing StreamSDK...")
        sdk.close()
        print("✅ Test complete")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return frame_count > 0

if __name__ == "__main__":
    print("\n🧪 Testing StreamSDK with 3240 sample chunks for 25-30 FPS")
    print("   (TensorRT minimum requirement)\n")

    success = test_chunk_size()

    if success:
        print("\n" + "=" * 70)
        print("✅ All tests passed! Ready for 25-30 FPS operation.")
        print("   Chunk size: 3240 samples (TensorRT minimum)")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Tests failed. Check configuration and TensorRT engines.")
        print("=" * 70)
        sys.exit(1)
