#!/usr/bin/env python3
"""
Simple speed test - just measure end-to-end pipeline throughput.
"""

import time
import sys
import librosa
import math
import numpy as np
from stream_pipeline_offline import StreamSDK

def main():
    print("="*70)
    print("SIMPLE THROUGHPUT TEST")
    print("="*70)

    # Initialize SDK
    print("\nInitializing SDK...")
    cfg_pkl = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
    data_root = "./checkpoints/ditto_trt_custom"

    SDK = StreamSDK(cfg_pkl, data_root)

    # Setup with dummy source
    print("Setting up pipeline...")

    # Create a simple test image
    import cv2
    test_img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    cv2.imwrite("test_img.jpg", test_img)

    SDK.setup("test_img.jpg", "speed_test_output.mp4")

    # Create short audio (5 seconds)
    print("Creating test audio (5 seconds)...")
    audio = np.zeros(16000 * 5, dtype=np.float32)
    num_frames = math.ceil(len(audio) / 16000 * 25)  # 125 frames
    print(f"Target frames: {num_frames}")

    SDK.setup_Nd(N_d=num_frames)

    # Process chunks
    print("\nProcessing audio chunks...")
    chunksize = (3, 5, 2)
    padding_samples = chunksize[0] * 640
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    chunk_hop = chunksize[1] * 640
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80

    chunk_count = 0
    chunk_start = time.time()

    for i in range(0, len(audio_padded), chunk_hop):
        audio_chunk = audio_padded[i:i + chunk_total_samples]
        if len(audio_chunk) < chunk_total_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, chunk_total_samples - len(audio_chunk)),
                mode="constant"
            )

        SDK.run_chunk(audio_chunk, chunksize)
        chunk_count += 1

    chunk_elapsed = time.time() - chunk_start
    print(f"✓ Submitted {chunk_count} chunks in {chunk_elapsed:.2f}s")
    print(f"  Chunk processing speed: {chunk_count/chunk_elapsed:.1f} chunks/s")

    # Signal end
    SDK.audio2motion_queue.put(None)

    # Monitor frame rendering
    print(f"\nWaiting for frame rendering...")
    print(f"Target: {num_frames} frames")
    print(f"\nProgress:")

    render_start = time.time()
    last_count = 0
    last_time = render_start

    while SDK.writer_pbar.n < num_frames:
        time.sleep(1)
        current = SDK.writer_pbar.n
        current_time = time.time()

        if current > last_count:
            # Calculate instantaneous FPS
            frames_rendered = current - last_count
            time_elapsed = current_time - last_time
            instant_fps = frames_rendered / time_elapsed if time_elapsed > 0 else 0

            # Calculate overall metrics
            total_elapsed = current_time - render_start
            overall_fps = current / total_elapsed if total_elapsed > 0 else 0
            remaining_frames = num_frames - current
            eta = remaining_frames / overall_fps if overall_fps > 0 else 0

            print(f"  [{current:3d}/{num_frames}] "
                  f"Overall: {overall_fps:.2f} fps | "
                  f"Instant: {instant_fps:.2f} fps | "
                  f"ETA: {eta:.0f}s")

            last_count = current
            last_time = current_time

        # Timeout after 2 minutes
        if current_time - render_start > 120:
            print(f"\n⚠ Timeout! Only rendered {current}/{num_frames} frames in 120s")
            break

    total_time = time.time() - render_start
    final_count = SDK.writer_pbar.n

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Frames rendered: {final_count}/{num_frames}")
    print(f"Total time: {total_time:.2f}s")

    if final_count > 0:
        avg_fps = final_count / total_time
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Time per frame: {total_time/final_count*1000:.0f} ms")

        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)

        if avg_fps >= 25:
            print("✓ REAL-TIME capable! (≥25 fps)")
        elif avg_fps >= 10:
            print("⚠ Moderate speed (10-25 fps)")
            print("  Real-time factor: {:.1f}x".format(avg_fps / 25))
        elif avg_fps >= 5:
            print("⚠ SLOW (5-10 fps)")
            print("  Real-time factor: {:.1f}x".format(avg_fps / 25))
        else:
            print("✗ VERY SLOW (<5 fps)")
            print("  Real-time factor: {:.1f}x".format(avg_fps / 25))
            print("\n  This is too slow for online/streaming mode!")
            print("  Likely bottleneck: LMDM diffusion model")
            print("  Recommendation: Reduce sampling_timesteps from 10 to 5")

    # Cleanup
    print("\n" + "="*70)
    print("Cleaning up...")
    SDK.stop_event.set()
    time.sleep(1)

if __name__ == "__main__":
    sys.exit(main())
