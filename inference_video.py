"""
Video-to-Video Inference Script for Ditto Talking Head

This script is optimized for using a video as the source (instead of an image).
It maintains the original video style while syncing lip movements to new audio.

Use Case: You have a neutral avatar video and want to make it speak with new audio.
"""

import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import argparse

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run_video_inference(
    SDK: StreamSDK,
    audio_path: str,
    source_video_path: str,
    output_path: str,
    more_kwargs: dict = None
):
    """
    Run inference with a video source.

    Args:
        SDK: Initialized StreamSDK
        audio_path: Path to input audio (.wav)
        source_video_path: Path to source video (.mp4, .avi, etc.)
        output_path: Path to output video (.mp4)
        more_kwargs: Additional configuration options
            - setup_kwargs: Parameters for SDK.setup()
            - run_kwargs: Parameters for SDK.setup_Nd() and audio processing
    """

    if more_kwargs is None:
        more_kwargs = {}

    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    # Video-specific default parameters (optimized for video sources)
    video_defaults = {
        # Source video smoothing - higher value for smoother transitions
        "smo_k_s": 13,

        # Eye control - typically False for videos to preserve original eye movements
        "eye_f0_mode": True,  # Use first frame eye state

        # Driving eye - let the model control eyes naturally
        "drive_eye": None,  # None = auto (False for video, True for image)

        # Maximum resolution
        "max_size": 1920,

        # Crop parameters - usually good defaults
        "crop_scale": 2.3,
        "crop_vx_ratio": 0,
        "crop_vy_ratio": -0.125,
        "crop_flag_do_rot": True,

        # Template frames - use multiple frames from source video
        "template_n_frames": -1,  # -1 = use all frames from source video
    }

    # Merge video defaults with user-provided setup_kwargs
    for key, value in video_defaults.items():
        if key not in setup_kwargs:
            setup_kwargs[key] = value

    print("=" * 60)
    print("VIDEO-TO-VIDEO INFERENCE")
    print("=" * 60)
    print(f"Source video: {source_video_path}")
    print(f"Audio file: {audio_path}")
    print(f"Output path: {output_path}")
    print(f"Video-specific parameters:")
    for key in video_defaults:
        print(f"  {key}: {setup_kwargs[key]}")
    print("=" * 60)

    # Setup SDK with video source
    SDK.setup(source_video_path, output_path, **setup_kwargs)

    # Load audio and calculate number of frames
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    print(f"\nAudio duration: {len(audio) / 16000:.2f} seconds")
    print(f"Generated frames: {num_f} frames (25 FPS)")
    print(f"Source video frames: {SDK.source_info_frames}")

    # Get fade and control info
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})

    # Setup frame count and control
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    # Process audio
    online_mode = SDK.online_mode
    if online_mode:
        print("\nProcessing in ONLINE mode (streaming)...")
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        print("\nProcessing in OFFLINE mode (batch)...")
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)

    print("Generating video frames...")
    SDK.close()

    # Add audio to the generated video
    print("\nMuxing audio with video...")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    print("\n" + "=" * 60)
    print(f"✓ Video generation complete!")
    print(f"✓ Output saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate talking head video from a source video and audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Basic usage with PyTorch model:
    python inference_video.py \\
        --source_video neutral_avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4

  2. High quality with TensorRT:
    python inference_video.py \\
        --data_root ./checkpoints/ditto_trt_Ampere_Plus \\
        --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output output.mp4 \\
        --sampling_timesteps 75 \\
        --overlap_v2 15

  3. With fade effects:
    python inference_video.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output output.mp4 \\
        --fade_in 25 \\
        --fade_out 25

  4. Fast preview (lower quality):
    python inference_video.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output preview.mp4 \\
        --sampling_timesteps 25 \\
        --max_size 512
        """
    )

    # Model configuration
    parser.add_argument("--data_root", type=str,
                       default="./checkpoints/ditto_pytorch",
                       help="Path to model directory")
    parser.add_argument("--cfg_pkl", type=str,
                       default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
                       help="Path to configuration pickle file")

    # Input/Output
    parser.add_argument("--source_video", type=str, required=True,
                       help="Path to source video (neutral avatar)")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to input audio (.wav)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output video (.mp4)")

    # Video-specific parameters
    parser.add_argument("--smo_k_s", type=int, default=13,
                       help="Source video smoothing kernel size (default: 13)")
    parser.add_argument("--template_n_frames", type=int, default=-1,
                       help="Number of frames to use from source video (-1 = all)")
    parser.add_argument("--drive_eye", type=str, default="auto",
                       choices=["auto", "true", "false"],
                       help="Control eye movements: auto (default for video type), true, or false")

    # Quality parameters
    parser.add_argument("--sampling_timesteps", type=int, default=50,
                       help="Diffusion sampling steps (25-100, higher=better quality, default: 50)")
    parser.add_argument("--overlap_v2", type=int, default=10,
                       help="Overlap frames for smooth transitions (5-20, default: 10)")
    parser.add_argument("--smo_k_d", type=int, default=3,
                       help="Motion smoothing kernel size (1-5, default: 3)")
    parser.add_argument("--max_size", type=int, default=1920,
                       help="Maximum resolution (default: 1920)")

    # Emotion and expression
    parser.add_argument("--emo", type=int, default=4,
                       help="Emotion value (0-7, default: 4 for neutral)")

    # Effects
    parser.add_argument("--fade_in", type=int, default=-1,
                       help="Fade in duration in frames (25 fps, -1 to disable)")
    parser.add_argument("--fade_out", type=int, default=-1,
                       help="Fade out duration in frames (25 fps, -1 to disable)")

    # Other
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        seed_everything(args.seed)

    # Initialize SDK
    print("Initializing Ditto SDK...")
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    # Prepare configuration
    setup_kwargs = {
        "smo_k_s": args.smo_k_s,
        "template_n_frames": args.template_n_frames,
        "sampling_timesteps": args.sampling_timesteps,
        "overlap_v2": args.overlap_v2,
        "smo_k_d": args.smo_k_d,
        "max_size": args.max_size,
        "emo": args.emo,
    }

    # Handle drive_eye parameter
    if args.drive_eye == "auto":
        setup_kwargs["drive_eye"] = None  # Auto-detect based on source type
    elif args.drive_eye == "true":
        setup_kwargs["drive_eye"] = True
    else:
        setup_kwargs["drive_eye"] = False

    run_kwargs = {
        "fade_in": args.fade_in,
        "fade_out": args.fade_out,
    }

    more_kwargs = {
        "setup_kwargs": setup_kwargs,
        "run_kwargs": run_kwargs,
    }

    # Run inference
    run_video_inference(
        SDK=SDK,
        audio_path=args.audio,
        source_video_path=args.source_video,
        output_path=args.output,
        more_kwargs=more_kwargs
    )


if __name__ == "__main__":
    main()
