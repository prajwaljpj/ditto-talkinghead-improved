#!/usr/bin/env python3
"""
Profile each pipeline stage to identify bottlenecks.
Patches the StreamSDK to add timing measurements.
"""

import time
import sys
import librosa
import math
import numpy as np
from collections import defaultdict, deque

def add_profiling(sdk):
    """Add profiling to all worker threads."""

    # Store timing data
    sdk.stage_times = {
        'audio2motion': deque(maxlen=100),
        'motion_stitch': deque(maxlen=100),
        'warp_f3d': deque(maxlen=100),
        'decode_f3d': deque(maxlen=100),
        'putback': deque(maxlen=100),
        'writer': deque(maxlen=100),
    }
    sdk.stage_counts = defaultdict(int)

    # Patch audio2motion worker
    original_a2m = sdk.audio2motion.__call__

    def timed_audio2motion(*args, **kwargs):
        start = time.time()
        result = original_a2m(*args, **kwargs)
        elapsed = time.time() - start
        sdk.stage_times['audio2motion'].append(elapsed)
        sdk.stage_counts['audio2motion'] += 1
        return result

    sdk.audio2motion.__call__ = timed_audio2motion

    # Patch motion_stitch
    original_stitch = sdk.motion_stitch.__call__

    def timed_stitch(*args, **kwargs):
        start = time.time()
        result = original_stitch(*args, **kwargs)
        elapsed = time.time() - start
        sdk.stage_times['motion_stitch'].append(elapsed)
        sdk.stage_counts['motion_stitch'] += 1
        return result

    sdk.motion_stitch.__call__ = timed_stitch

    # Patch warp_f3d
    original_warp = sdk.warp_f3d.__call__

    def timed_warp(*args, **kwargs):
        start = time.time()
        result = original_warp(*args, **kwargs)
        elapsed = time.time() - start
        sdk.stage_times['warp_f3d'].append(elapsed)
        sdk.stage_counts['warp_f3d'] += 1
        return result

    sdk.warp_f3d.__call__ = timed_warp

    # Patch decode_f3d
    original_decode = sdk.decode_f3d.__call__

    def timed_decode(*args, **kwargs):
        start = time.time()
        result = original_decode(*args, **kwargs)
        elapsed = time.time() - start
        sdk.stage_times['decode_f3d'].append(elapsed)
        sdk.stage_counts['decode_f3d'] += 1
        return result

    sdk.decode_f3d.__call__ = timed_decode

    # Patch putback
    original_putback = sdk.putback.__call__

    def timed_putback(*args, **kwargs):
        start = time.time()
        result = original_putback(*args, **kwargs)
        elapsed = time.time() - start
        sdk.stage_times['putback'].append(elapsed)
        sdk.stage_counts['putback'] += 1
        return result

    sdk.putback.__call__ = timed_putback

    # Patch writer
    original_writer = sdk.writer.__call__

    def timed_writer(*args, **kwargs):
        start = time.time()
        result = original_writer(*args, **kwargs)
        elapsed = time.time() - start
        sdk.stage_times['writer'].append(elapsed)
        sdk.stage_counts['writer'] += 1
        return result

    sdk.writer.__call__ = timed_writer


def print_profile_stats(sdk):
    """Print profiling statistics."""
    print("\n" + "="*70)
    print("PIPELINE STAGE PROFILING")
    print("="*70)

    stages = ['audio2motion', 'motion_stitch', 'warp_f3d', 'decode_f3d', 'putback', 'writer']

    for stage in stages:
        times = list(sdk.stage_times[stage])
        count = sdk.stage_counts[stage]

        if not times:
            print(f"\n{stage}:")
            print(f"  Count: {count}")
            print(f"  Status: No timing data (not called yet or too fast)")
            continue

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        print(f"\n{stage}:")
        print(f"  Count: {count}")
        print(f"  Avg time: {avg_time*1000:.2f} ms ({fps:.2f} fps)")
        print(f"  Min time: {min_time*1000:.2f} ms")
        print(f"  Max time: {max_time*1000:.2f} ms")

        # Highlight slow stages
        if avg_time > 0.1:  # > 100ms = slow
            print(f"  ⚠️  SLOW STAGE! (target: <40ms for 25fps)")

    # Identify bottleneck
    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS")
    print("="*70)

    avg_times = {}
    for stage in stages:
        times = list(sdk.stage_times[stage])
        if times:
            avg_times[stage] = sum(times) / len(times)

    if avg_times:
        slowest = max(avg_times.items(), key=lambda x: x[1])
        print(f"\nSlowest stage: {slowest[0]}")
        print(f"Average time: {slowest[1]*1000:.2f} ms ({1.0/slowest[1]:.2f} fps)")
        print(f"\nThis stage is limiting your throughput!")

        # Provide specific advice
        advice = {
            'audio2motion': "LMDM diffusion model is slow. Try reducing sampling_timesteps.",
            'motion_stitch': "Motion stitching overhead. This is usually fast, check GPU sync.",
            'warp_f3d': "TensorRT warp network is slow. Check GPU utilization and batch size.",
            'decode_f3d': "TensorRT decoder is slow. Check GPU utilization.",
            'putback': "Image compositing is slow. Check CPU usage.",
            'writer': "Video writing is slow. Check disk speed or use faster codec.",
        }

        if slowest[0] in advice:
            print(f"\nOptimization advice:")
            print(f"  {advice[slowest[0]]}")


def main():
    import argparse
    from stream_pipeline_offline import StreamSDK

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_trt_custom")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    parser.add_argument("--test_frames", type=int, default=50,
                       help="Number of frames to process for profiling")

    args = parser.parse_args()

    print("="*70)
    print("PIPELINE STAGE PROFILER")
    print("="*70)

    # Initialize
    print("\n[1/4] Initializing SDK...")
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    print("[2/4] Setting up pipeline...")
    SDK.setup(args.source, "profile_test_output.mp4")

    # Add profiling after setup (when writer exists)
    print("[3/4] Adding profiling hooks...")
    add_profiling(SDK)

    # Load audio
    print("[4/4] Loading audio...")
    audio, sr = librosa.core.load(args.audio, sr=16000)
    num_frames = min(args.test_frames, math.ceil(len(audio) / 16000 * 25))
    print(f"    Will process {num_frames} frames for profiling")

    SDK.setup_Nd(N_d=num_frames)

    # Process chunks
    print(f"\nProcessing frames (this will take ~{num_frames}s at 1fps)...")

    chunksize = (3, 5, 2)
    padding_samples = chunksize[0] * 640
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    chunk_hop = chunksize[1] * 640
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80

    # Calculate chunks needed
    chunks_needed = math.ceil(num_frames / chunksize[1]) + 2

    chunk_count = 0
    for i in range(0, len(audio_padded), chunk_hop):
        if chunk_count >= chunks_needed:
            break

        audio_chunk = audio_padded[i:i + chunk_total_samples]
        if len(audio_chunk) < chunk_total_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, chunk_total_samples - len(audio_chunk)),
                mode="constant"
            )

        SDK.run_chunk(audio_chunk, chunksize)
        chunk_count += 1

        # Print progress every 5 chunks
        if chunk_count % 5 == 0:
            print(f"    Submitted {chunk_count} chunks, rendered ~{SDK.writer_pbar.n} frames...")

    print(f"    Submitted {chunk_count} chunks")

    # Wait for frames to be rendered
    print(f"\n    Waiting for pipeline to render frames...")
    SDK.audio2motion_queue.put(None)

    last_count = SDK.writer_pbar.n
    stall_count = 0

    while SDK.writer_pbar.n < num_frames:
        time.sleep(2)
        current = SDK.writer_pbar.n
        print(f"    Progress: {current}/{num_frames} frames...", flush=True)

        if current == last_count:
            stall_count += 1
            if stall_count > 10:  # 20 seconds
                print(f"\n    Stopping early (stalled at {current} frames)")
                break
        else:
            stall_count = 0

        last_count = current

    # Print statistics
    print_profile_stats(SDK)

    # Cleanup
    print("\n" + "="*70)
    print("Cleaning up...")
    SDK.stop_event.set()
    time.sleep(1)

    print("\n" + "="*70)
    print("✓ Profiling complete!")
    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
