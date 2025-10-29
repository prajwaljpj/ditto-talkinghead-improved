#!/usr/bin/env python3
"""
Monitor pipeline performance in real-time to identify bottlenecks.
"""

import sys
import time
import threading
import librosa
import math
import numpy as np

def monitor_pipeline(sdk, duration=5):
    """Monitor pipeline queues and progress in real-time."""
    start_time = time.time()

    while time.time() - start_time < duration:
        print(f"\n{'='*70}")
        print(f"Pipeline Status [{time.time()-start_time:.1f}s]")
        print(f"{'='*70}")

        # Queue sizes
        print(f"Queue Sizes:")
        print(f"  audio2motion_queue:   {sdk.audio2motion_queue.qsize():4d}")
        print(f"  motion_stitch_queue:  {sdk.motion_stitch_queue.qsize():4d}")
        print(f"  warp_f3d_queue:       {sdk.warp_f3d_queue.qsize():4d}")
        print(f"  decode_f3d_queue:     {sdk.decode_f3d_queue.qsize():4d}")
        print(f"  putback_queue:        {sdk.putback_queue.qsize():4d}")
        print(f"  writer_queue:         {sdk.writer_queue.qsize():4d}")

        # Thread status
        thread_names = [
            "audio2motion",
            "motion_stitch",
            "warp_f3d",
            "decode_f3d",
            "putback",
            "writer",
        ]

        alive_count = sum(1 for t in sdk.thread_list if t.is_alive())
        print(f"\nThreads: {alive_count}/{len(sdk.thread_list)} alive")

        # Check for exceptions
        if sdk.worker_exception:
            print(f"\n⚠ Worker Exception: {sdk.worker_exception}")
            return False

        time.sleep(1)

    return True


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
    parser.add_argument("--num_chunks", type=int, default=20,
                       help="Number of chunks to process for testing")

    args = parser.parse_args()

    print("="*70)
    print("PIPELINE PERFORMANCE MONITOR")
    print("="*70)

    # Initialize
    print("\n[1] Initializing SDK...")
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    print("[2] Setting up pipeline...")
    SDK.setup(args.source, "test_output.mp4")
    print(f"    Online mode: {SDK.online_mode}")

    # Load audio
    print("[3] Loading audio...")
    audio, sr = librosa.core.load(args.audio, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    SDK.setup_Nd(N_d=num_frames)
    print(f"    Audio: {len(audio)/16000:.2f}s, {num_frames} frames")

    # Prepare chunks
    chunksize = (3, 5, 2)
    padding_samples = chunksize[0] * 640
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    chunk_hop = chunksize[1] * 640
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80

    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_pipeline,
        args=(SDK, 999999),  # Monitor until interrupted
        daemon=True
    )
    monitor_thread.start()

    # Process chunks
    print(f"\n[4] Processing {args.num_chunks} chunks...")
    print("    (Watch the queue sizes below)\n")

    chunk_count = 0
    for i in range(0, len(audio_padded), chunk_hop):
        if chunk_count >= args.num_chunks:
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

        # Small delay to let pipeline process
        time.sleep(0.01)

    print(f"\n[5] Finished submitting {chunk_count} chunks")
    print("    Waiting 10 seconds for pipeline to process...\n")

    # Wait and observe
    time.sleep(10)

    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    print(f"Queue Sizes:")
    print(f"  audio2motion_queue:   {SDK.audio2motion_queue.qsize():4d}")
    print(f"  motion_stitch_queue:  {SDK.motion_stitch_queue.qsize():4d}")
    print(f"  warp_f3d_queue:       {SDK.warp_f3d_queue.qsize():4d}")
    print(f"  decode_f3d_queue:     {SDK.decode_f3d_queue.qsize():4d}")
    print(f"  putback_queue:        {SDK.putback_queue.qsize():4d}")
    print(f"  writer_queue:         {SDK.writer_queue.qsize():4d}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Identify bottleneck
    queues = {
        "audio2motion": SDK.audio2motion_queue.qsize(),
        "motion_stitch": SDK.motion_stitch_queue.qsize(),
        "warp_f3d": SDK.warp_f3d_queue.qsize(),
        "decode_f3d": SDK.decode_f3d_queue.qsize(),
        "putback": SDK.putback_queue.qsize(),
        "writer": SDK.writer_queue.qsize(),
    }

    max_queue = max(queues.items(), key=lambda x: x[1])

    if max_queue[1] > 50:
        print(f"⚠ BOTTLENECK DETECTED: {max_queue[0]}_queue is full ({max_queue[1]} items)")
        print(f"  The stage AFTER {max_queue[0]} is too slow!")
        print(f"\n  Pipeline stages:")
        print(f"    1. audio2motion (generates motion)")
        print(f"    2. motion_stitch (stitches motion)")
        print(f"    3. warp_f3d (warps 3D features) ← Usually slowest")
        print(f"    4. decode_f3d (decodes to image)")
        print(f"    5. putback (composites image)")
        print(f"    6. writer (writes to disk)")
    else:
        print("✓ No obvious bottleneck (all queues < 50)")

    print("\n[Ctrl+C to exit]")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nCleaning up...")
        SDK.stop_event.set()


if __name__ == "__main__":
    sys.exit(main())
