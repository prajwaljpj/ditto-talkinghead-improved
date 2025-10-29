#!/usr/bin/env python3
"""
Diagnose which worker thread is causing the pipeline hang.
This patches the StreamSDK to add monitoring.
"""

import sys
import time
import threading
from pathlib import Path

def monitor_threads(sdk, timeout=10):
    """Monitor worker threads and report which are alive/stuck."""
    print(f"\n{'='*70}")
    print(f"Monitoring worker threads for {timeout}s...")
    print(f"{'='*70}")

    thread_names = [
        "audio2motion_worker",
        "motion_stitch_worker",
        "warp_f3d_worker",
        "decode_f3d_worker",
        "putback_worker",
        "writer_worker",
    ]

    start_time = time.time()

    while time.time() - start_time < timeout:
        print(f"\n[{time.time()-start_time:.1f}s] Thread Status:")

        for i, thread in enumerate(sdk.thread_list):
            name = thread_names[i] if i < len(thread_names) else f"thread_{i}"
            status = "RUNNING" if thread.is_alive() else "FINISHED"
            print(f"  {name}: {status}")

        # Check queue sizes
        print(f"\n[{time.time()-start_time:.1f}s] Queue Sizes:")
        print(f"  audio2motion_queue: {sdk.audio2motion_queue.qsize()}")
        print(f"  motion_stitch_queue: {sdk.motion_stitch_queue.qsize()}")
        print(f"  warp_f3d_queue: {sdk.warp_f3d_queue.qsize()}")
        print(f"  decode_f3d_queue: {sdk.decode_f3d_queue.qsize()}")
        print(f"  putback_queue: {sdk.putback_queue.qsize()}")
        print(f"  writer_queue: {sdk.writer_queue.qsize()}")

        # Check if all threads finished
        if not any(t.is_alive() for t in sdk.thread_list):
            print("\n✓ All threads finished!")
            return True

        time.sleep(2)

    print(f"\n✗ Timeout! Still waiting for threads to finish")
    print("\nStuck threads:")
    for i, thread in enumerate(sdk.thread_list):
        if thread.is_alive():
            name = thread_names[i] if i < len(thread_names) else f"thread_{i}"
            print(f"  - {name}")

    return False


def main():
    import argparse
    import librosa
    import math
    import numpy as np
    from stream_pipeline_offline import StreamSDK

    parser = argparse.ArgumentParser(description="Diagnose pipeline hang")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_output.mp4")
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_trt_custom")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")

    args = parser.parse_args()

    print("="*70)
    print("PIPELINE HANG DIAGNOSTIC")
    print("="*70)

    # Initialize SDK
    print("\n[1] Initializing SDK...")
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    print("✓ SDK initialized")

    # Setup
    print("\n[2] Setting up pipeline...")
    SDK.setup(args.source, args.output)
    print(f"✓ Online mode: {SDK.online_mode}")

    # Load audio
    print("\n[3] Loading audio...")
    audio, sr = librosa.core.load(args.audio, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    print(f"✓ Audio: {len(audio)/16000:.2f}s, {num_frames} frames")

    SDK.setup_Nd(N_d=num_frames)

    # Process a few chunks only (to speed up diagnosis)
    print("\n[4] Processing 10 test chunks...")
    chunksize = (3, 5, 2)

    padding_samples = chunksize[0] * 640
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    chunk_hop = chunksize[1] * 640
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80

    chunk_count = 0
    max_chunks = 10  # Only process 10 chunks for quick diagnosis

    for i in range(0, len(audio_padded), chunk_hop):
        if chunk_count >= max_chunks:
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
        print(f"  Chunk {chunk_count}/{max_chunks}")

    print(f"✓ Processed {chunk_count} chunks")

    # Check for worker exceptions before closing
    if SDK.worker_exception:
        print(f"\n✗ Worker exception detected: {SDK.worker_exception}")
        import traceback
        traceback.print_exception(type(SDK.worker_exception),
                                   SDK.worker_exception,
                                   SDK.worker_exception.__traceback__)
        return 1

    # Now try to close and monitor what happens
    print("\n[5] Closing pipeline (this is where it usually hangs)...")
    print("    Putting None in audio2motion_queue...")
    SDK.audio2motion_queue.put(None)

    # Monitor threads
    success = monitor_threads(SDK, timeout=30)

    if not success:
        print("\n" + "="*70)
        print("DIAGNOSIS: Pipeline deadlock detected")
        print("="*70)
        print("\nPossible causes:")
        print("1. A worker thread crashed silently")
        print("2. Queues are full and blocking")
        print("3. Worker not propagating None signal")
        print("\nTry adding debug prints to worker threads in stream_pipeline_online.py")
        return 1

    # If successful, finish normally
    print("\n✓ Threads finished, finalizing...")
    SDK.writer.close()
    SDK.writer_pbar.close()

    print("\n" + "="*70)
    print("✓ Diagnostic complete - no hang detected!")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
