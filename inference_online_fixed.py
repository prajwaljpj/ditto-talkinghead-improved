"""
Fixed online inference that waits for pipeline to process all frames.
"""

import librosa
import math
import os
import sys
import time
import numpy as np
from tqdm import tqdm

def run_online_inference_fixed(
    cfg_pkl: str,
    data_root: str,
    source_path: str,
    audio_path: str,
    output_path: str,
    chunksize: tuple = (3, 5, 2)
):
    """
    Run inference in online/streaming mode with proper pipeline draining.
    """

    print("=" * 70)
    print("ONLINE MODE INFERENCE - FIXED")
    print("=" * 70)

    # Initialize SDK
    print("[1/6] Initializing SDK...")
    from stream_pipeline_offline import StreamSDK
    SDK = StreamSDK(cfg_pkl, data_root)
    print(f"✓ SDK initialized")

    # Setup
    print("[2/6] Setting up pipeline...")
    SDK.setup(source_path, output_path)
    print(f"✓ Online mode: {SDK.online_mode}")

    # Load audio
    print("[3/6] Loading audio...")
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    print(f"✓ Audio: {len(audio)/16000:.2f}s, {num_frames} frames expected")

    SDK.setup_Nd(N_d=num_frames)

    # Process audio in chunks
    print(f"[4/6] Processing audio chunks...")

    padding_samples = chunksize[0] * 640
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    chunk_hop = chunksize[1] * 640
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80

    chunk_count = 0
    total_chunks = len(range(0, len(audio_padded), chunk_hop))

    print(f"     Total chunks: {total_chunks}")
    print(f"     Submitting chunks to pipeline...")

    pbar = tqdm(total=total_chunks, desc="Chunks", unit="chunk")

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
        pbar.update(1)

    pbar.close()
    print(f"✓ Submitted {chunk_count} chunks")

    # Signal end of input
    print("[5/6] Waiting for pipeline to process frames...")
    print(f"     Expected frames: {num_frames}")
    print(f"     This may take a while for long videos...")

    SDK.audio2motion_queue.put(None)

    # Wait for frames to be written
    # The writer_pbar shows progress
    print(f"\n     Frame rendering progress:")

    # Monitor progress
    last_count = 0
    stall_count = 0
    start_wait = time.time()

    while True:
        time.sleep(2)

        # Check current frame count
        current_count = SDK.writer_pbar.n

        # Show progress
        if current_count > 0:
            progress_pct = (current_count / num_frames) * 100
            elapsed = time.time() - start_wait
            fps = current_count / elapsed if elapsed > 0 else 0
            eta = (num_frames - current_count) / fps if fps > 0 else 0

            print(f"     Rendered {current_count}/{num_frames} frames "
                  f"({progress_pct:.1f}%) [{fps:.2f} fps, ETA: {eta:.0f}s]", flush=True)

        # Check if done
        if current_count >= num_frames:
            print(f"     ✓ All {num_frames} frames rendered!")
            break

        # Check for stall
        if current_count == last_count:
            stall_count += 1
            if stall_count > 60:  # 2 minutes of no progress
                print(f"\n     ⚠ Pipeline appears stalled at {current_count} frames")
                print(f"     Queue sizes:")
                print(f"       audio2motion: {SDK.audio2motion_queue.qsize()}")
                print(f"       motion_stitch: {SDK.motion_stitch_queue.qsize()}")
                print(f"       warp_f3d: {SDK.warp_f3d_queue.qsize()}")
                print(f"       decode_f3d: {SDK.decode_f3d_queue.qsize()}")
                print(f"       putback: {SDK.putback_queue.qsize()}")
                print(f"       writer: {SDK.writer_queue.qsize()}")

                # Check for exceptions
                if SDK.worker_exception:
                    print(f"\n     ✗ Worker exception: {SDK.worker_exception}")
                    raise SDK.worker_exception

                print(f"\n     Continuing to wait...")
                stall_count = 0
        else:
            stall_count = 0

        last_count = current_count

        # Check for worker exceptions
        if SDK.worker_exception:
            print(f"\n✗ Worker exception: {SDK.worker_exception}")
            raise SDK.worker_exception

    # Now close (threads should finish quickly since all frames are rendered)
    print(f"[6/6] Finalizing video...")

    # Wait for threads
    for thread in SDK.thread_list:
        thread.join(timeout=10)

    # Close writer
    SDK.writer.close()
    SDK.writer_pbar.close()

    # Add audio track
    print(f"     Muxing audio...")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    print("\n" + "=" * 70)
    print("✓ Online inference complete!")
    print(f"✓ Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Ditto online inference (fixed version)"
    )

    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_trt_custom")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--chunksize", type=int, nargs=3, default=[3, 5, 2])

    args = parser.parse_args()

    try:
        run_online_inference_fixed(
            cfg_pkl=args.cfg_pkl,
            data_root=args.data_root,
            source_path=args.source,
            audio_path=args.audio,
            output_path=args.output,
            chunksize=tuple(args.chunksize)
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
