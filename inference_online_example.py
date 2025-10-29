"""
Online Mode Inference Example

Demonstrates how to run Ditto in online/streaming mode.
"""

import librosa
import math
import os
import numpy as np

from stream_pipeline_offline import StreamSDK  # Yes, uses same SDK


def run_online_inference(
    cfg_pkl: str,
    data_root: str,
    source_path: str,
    audio_path: str,
    output_path: str,
    chunksize: tuple = (3, 5, 2)
):
    """
    Run inference in online/streaming mode.

    Args:
        cfg_pkl: Path to ONLINE config (must be online config!)
        data_root: Path to model directory
        source_path: Path to source image/video
        audio_path: Path to audio file
        output_path: Path to output video
        chunksize: (past_context, current_chunk, future_context)
                   Default (3,5,2) = 3 past + 5 current + 2 future frames
    """

    print("=" * 70)
    print("ONLINE MODE INFERENCE")
    print("=" * 70)
    print(f"Source: {source_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunksize}")
    print("=" * 70)

    # Initialize SDK
    print("\n[1/5] Initializing SDK with online config...")
    SDK = StreamSDK(cfg_pkl, data_root)

    # Setup (will detect online mode from config)
    print("[2/5] Setting up pipeline...")
    SDK.setup(source_path, output_path)

    # Verify online mode is enabled
    if not SDK.online_mode:
        print("WARNING: Config doesn't enable online mode!")
        print("Make sure you're using an '_online.pkl' config file")

    print(f"      Online mode: {SDK.online_mode}")

    # Load audio
    print("[3/5] Loading audio...")
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)
    print(f"      Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"      Output frames: {num_frames}")

    # Setup frame count
    SDK.setup_Nd(N_d=num_frames)

    # Process audio in chunks (online mode)
    print("[4/5] Processing audio in streaming chunks...")

    # Add padding for context (past frames)
    padding_samples = chunksize[0] * 640  # 3 frames * 640 samples/frame
    audio_padded = np.concatenate([
        np.zeros((padding_samples,), dtype=np.float32),
        audio
    ], axis=0)

    # Calculate chunk parameters
    chunk_hop = chunksize[1] * 640  # Current chunk size in samples
    chunk_total_samples = int(sum(chunksize) * 0.04 * 16000) + 80  # Total chunk window

    print(f"      Chunk hop: {chunk_hop} samples ({chunksize[1]} frames)")
    print(f"      Chunk window: {chunk_total_samples} samples ({sum(chunksize)} frames)")

    # Process chunks
    chunk_count = 0
    for i in range(0, len(audio_padded), chunk_hop):
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
        SDK.run_chunk(audio_chunk, chunksize)
        chunk_count += 1

        if chunk_count % 10 == 0:
            print(f"      Processed {chunk_count} chunks...")

    print(f"      Total chunks processed: {chunk_count}")

    # Finalize
    print("[5/5] Finalizing video...")
    SDK.close()

    # Add audio track
    print("      Muxing audio...")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    print("\n" + "=" * 70)
    print("✓ Online inference complete!")
    print(f"✓ Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Ditto inference in online/streaming mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Basic online inference:
    python inference_online_example.py \\
        --source image.png \\
        --audio speech.wav \\
        --output result.mp4

  2. With TensorRT (faster):
    python inference_online_example.py \\
        --data_root ./checkpoints/ditto_trt_Ampere_Plus \\
        --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl \\
        --source image.png \\
        --audio speech.wav \\
        --output result.mp4

  3. Custom chunk size:
    python inference_online_example.py \\
        --source image.png \\
        --audio speech.wav \\
        --output result.mp4 \\
        --chunksize 3 5 2

Chunk Size Explanation:
  --chunksize 3 5 2 means:
    3 frames: Past context (history)
    5 frames: Current chunk to process
    2 frames: Future context (lookahead)

  Total window: 10 frames (0.4 seconds)
  Output per chunk: 5 frames (current)

  Lower values = Lower latency, but may affect quality
  Higher values = Better quality, but higher latency
        """
    )

    # Model configuration
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_pytorch",
                       help="Path to model directory")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
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
    run_online_inference(
        cfg_pkl=args.cfg_pkl,
        data_root=args.data_root,
        source_path=args.source,
        audio_path=args.audio,
        output_path=args.output,
        chunksize=tuple(args.chunksize)
    )
