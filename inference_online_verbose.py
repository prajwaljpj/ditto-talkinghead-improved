"""
Verbose version of online inference with detailed logging.
Use this to identify exactly where the process hangs.
"""

import librosa
import math
import os
import sys
import time
import numpy as np

def log(msg, level="INFO"):
    """Print with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

# Use patched version with better timeout handling
USE_PATCHED = True

def run_online_inference_verbose(
    cfg_pkl: str,
    data_root: str,
    source_path: str,
    audio_path: str,
    output_path: str,
    chunksize: tuple = (3, 5, 2)
):
    """
    Run inference in online/streaming mode with verbose logging.
    """

    log("="*70)
    log("ONLINE MODE INFERENCE - VERBOSE")
    log("="*70)
    log(f"Source: {source_path}")
    log(f"Audio: {audio_path}")
    log(f"Output: {output_path}")
    log(f"Chunk size: {chunksize}")
    log(f"Config: {cfg_pkl}")
    log(f"Data root: {data_root}")
    log("="*70)

    # Step 1: Import SDK
    log("Importing StreamSDK module...")
    start = time.time()

    if USE_PATCHED:
        log("Using PATCHED version with timeout handling", "INFO")
        from stream_pipeline_online_patched import StreamSDK
    else:
        from stream_pipeline_offline import StreamSDK

    log(f"Import complete ({time.time()-start:.2f}s)")

    # Step 2: Initialize SDK
    log("Initializing SDK (loading TensorRT engines)...")
    log("NOTE: This step can take 30-120s depending on engines")
    log("If it hangs here for >5 minutes, press Ctrl+C and run:")
    log("  python test_trt_engines.py")
    start = time.time()

    try:
        SDK = StreamSDK(cfg_pkl, data_root)
        log(f"SDK initialized ({time.time()-start:.2f}s)", "SUCCESS")
    except Exception as e:
        log(f"SDK initialization failed: {e}", "ERROR")
        raise

    # Step 3: Setup
    log("Setting up pipeline...")
    start = time.time()
    SDK.setup(source_path, output_path)
    log(f"Pipeline setup complete ({time.time()-start:.2f}s)")

    # Verify online mode
    if not SDK.online_mode:
        log("WARNING: Config doesn't enable online mode!", "WARN")
        log("Make sure you're using an '_online.pkl' config file", "WARN")
    else:
        log(f"Online mode confirmed: {SDK.online_mode}", "SUCCESS")

    # Step 4: Load audio
    log("Loading audio...")
    start = time.time()
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    log(f"Audio loaded: {len(audio)/16000:.2f}s, {num_frames} frames ({time.time()-start:.2f}s)")

    # Setup frame count
    SDK.setup_Nd(N_d=num_frames)

    # Step 5: Process audio in chunks
    log("Processing audio in streaming chunks...")

    # Add padding for context
    padding_samples = chunksize[0] * 640
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    chunk_hop = chunksize[1] * 640
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80

    log(f"Chunk parameters:")
    log(f"  - Hop: {chunk_hop} samples ({chunksize[1]} frames)")
    log(f"  - Window: {chunk_total_samples} samples ({sum(chunksize)} frames)")
    log(f"  - Total chunks: {len(range(0, len(audio_padded), chunk_hop))}")

    # Process chunks
    chunk_count = 0
    total_chunks = len(range(0, len(audio_padded), chunk_hop))
    start_processing = time.time()
    last_log_time = start_processing

    for i in range(0, len(audio_padded), chunk_hop):
        chunk_start = time.time()

        # Extract chunk
        audio_chunk = audio_padded[i:i + chunk_total_samples]

        # Pad last chunk if needed
        if len(audio_chunk) < chunk_total_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, chunk_total_samples - len(audio_chunk)),
                mode="constant"
            )

        # Process chunk
        try:
            SDK.run_chunk(audio_chunk, chunksize)
            chunk_count += 1

            # Log progress every 5 chunks or every 2 seconds
            chunk_time = time.time() - chunk_start
            current_time = time.time()

            if chunk_count % 5 == 0 or (current_time - last_log_time) >= 2.0:
                elapsed = current_time - start_processing
                chunks_per_sec = chunk_count / elapsed if elapsed > 0 else 0
                eta = (total_chunks - chunk_count) / chunks_per_sec if chunks_per_sec > 0 else 0

                log(f"Processed {chunk_count}/{total_chunks} chunks "
                    f"({chunk_count*100/total_chunks:.1f}%) "
                    f"[{chunks_per_sec:.2f} chunks/s, ETA: {eta:.1f}s]")
                last_log_time = current_time

            # Check if worker threads had exceptions
            if SDK.worker_exception:
                log(f"Worker thread exception detected: {SDK.worker_exception}", "ERROR")
                raise SDK.worker_exception

        except Exception as e:
            log(f"Error processing chunk {chunk_count}: {e}", "ERROR")
            raise

    total_processing_time = time.time() - start_processing
    log(f"All chunks processed in {total_processing_time:.2f}s "
        f"({chunk_count/total_processing_time:.2f} chunks/s)", "SUCCESS")

    # Step 6: Finalize
    log("Finalizing video (waiting for pipeline to flush)...")
    start = time.time()

    if USE_PATCHED:
        SDK.close(timeout=120)  # 2 minute timeout for patched version
    else:
        SDK.close()

    log(f"Pipeline closed ({time.time()-start:.2f}s)")

    # Step 7: Add audio track
    log("Muxing audio track with ffmpeg...")
    start = time.time()
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    ret = os.system(cmd)

    if ret != 0:
        log(f"ffmpeg returned error code: {ret}", "WARN")
    else:
        log(f"Audio muxed ({time.time()-start:.2f}s)")

    log("="*70)
    log("✓ Online inference complete!", "SUCCESS")
    log(f"✓ Output: {output_path}")
    log("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Ditto online inference with verbose logging"
    )

    # Model configuration
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_trt_custom",
                       help="Path to TensorRT model directory")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                       help="Path to ONLINE config file")

    # Input/Output
    parser.add_argument("--source", type=str, required=True,
                       help="Path to source image or video")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to audio file (.wav)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output video (.mp4)")

    # Online mode parameters
    parser.add_argument("--chunksize", type=int, nargs=3, default=[3, 5, 2],
                       metavar=("PAST", "CURRENT", "FUTURE"),
                       help="Chunk size (past, current, future) frames")

    args = parser.parse_args()

    # Run online inference
    try:
        run_online_inference_verbose(
            cfg_pkl=args.cfg_pkl,
            data_root=args.data_root,
            source_path=args.source,
            audio_path=args.audio,
            output_path=args.output,
            chunksize=tuple(args.chunksize)
        )
        sys.exit(0)
    except KeyboardInterrupt:
        log("\nInterrupted by user", "WARN")
        sys.exit(130)
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
