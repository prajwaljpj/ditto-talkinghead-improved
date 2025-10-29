"""
Video Inference with Configurable Blending Control

Control how much of the rendered face vs original face is used in the output.

Blend Modes:
- 'blend': Default smooth blending (50% center, feathered edges)
- 'strong': More rendered face (75% rendered, 25% original)
- 'replace': Full replacement (100% rendered, 0% original)
- 'weak': More original face (25% rendered, 75% original)
- 'custom': Custom percentage (specify with --blend_alpha)
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


def run_with_blend_control(
    SDK: StreamSDK,
    audio_path: str,
    source_path: str,
    output_path: str,
    blend_mode: str = 'blend',
    blend_alpha: float = None,
    more_kwargs: dict = None
):
    """
    Run inference with configurable blending.

    Args:
        SDK: Initialized StreamSDK
        audio_path: Path to audio
        source_path: Path to source image/video
        output_path: Path to output video
        blend_mode: 'blend', 'strong', 'replace', 'weak', or 'custom'
        blend_alpha: Alpha for 'custom' mode (0.0=all original, 1.0=all rendered)
        more_kwargs: Additional configuration
    """

    if more_kwargs is None:
        more_kwargs = {}

    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    # Video defaults
    video_defaults = {
        "smo_k_s": 13,
        "eye_f0_mode": True,
        "drive_eye": None,
        "max_size": 1920,
        "crop_scale": 2.3,
        "crop_vx_ratio": 0,
        "crop_vy_ratio": -0.125,
        "crop_flag_do_rot": True,
        "template_n_frames": -1,
    }

    for key, value in video_defaults.items():
        if key not in setup_kwargs:
            setup_kwargs[key] = value

    print("=" * 70)
    print("VIDEO INFERENCE WITH BLEND CONTROL")
    print("=" * 70)
    print(f"Source: {source_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")
    print(f"\nBlend Mode: {blend_mode}")
    if blend_mode == 'custom':
        print(f"Custom Alpha: {blend_alpha} (0.0=original, 1.0=rendered)")
    print("=" * 70)

    # Replace PutBack with configurable version
    from core.atomic_components.putback_configurable import get_putback_by_mode
    SDK.putback = get_putback_by_mode(
        blend_mode=blend_mode,
        custom_alpha=blend_alpha,
    )

    # Setup
    SDK.setup(source_path, output_path, **setup_kwargs)

    # Load audio
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    print(f"\nAudio duration: {len(audio) / 16000:.2f} seconds")
    print(f"Output frames: {num_f} frames")

    # Setup frame count
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    # Process audio
    online_mode = SDK.online_mode
    if online_mode:
        print("\nProcessing in ONLINE mode...")
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        print("\nProcessing in OFFLINE mode...")
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)

    print("Generating video...")
    SDK.close()

    # Add audio
    print("Muxing audio...")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    print("\n" + "=" * 70)
    print("✓ Video generation complete!")
    print(f"✓ Output: {output_path}")
    print(f"✓ Blend mode used: {blend_mode}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate talking video with configurable face blending",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Blend Modes:

  blend (default):  Smooth feathered blending
                    Center: 50% rendered, 50% original
                    Edges: Gradual transition
                    Best for: Natural looking results

  strong:           More rendered face
                    Center: 90% rendered, 10% original
                    Good for: Stronger lip-sync effect

  replace:          Full replacement
                    100% rendered face
                    Good for: Maximum lip-sync fidelity
                    Note: May look less natural at edges

  weak:             More original face
                    Center: 50% rendered, 50% original
                    Good for: Subtle lip-sync

  custom:           Custom percentage
                    Use --blend_alpha to specify
                    0.0 = 100% original
                    0.5 = 50/50 blend
                    1.0 = 100% rendered

Examples:

  1. Default blending:
    python inference_video_blend_control.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4

  2. Strong rendered face (90%):
    python inference_video_blend_control.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result_strong.mp4 \\
        --blend_mode strong

  3. Full replacement (100%):
    python inference_video_blend_control.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result_replace.mp4 \\
        --blend_mode replace

  4. Custom 70% rendered:
    python inference_video_blend_control.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result_custom.mp4 \\
        --blend_mode custom \\
        --blend_alpha 0.7
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
                       help="Path to source image/video")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to input audio (.wav)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output video (.mp4)")

    # Blending control
    parser.add_argument("--blend_mode", type=str, default='blend',
                       choices=['blend', 'strong', 'replace', 'weak', 'custom'],
                       help="Blending mode (default: blend)")
    parser.add_argument("--blend_alpha", type=float, default=None,
                       help="Custom alpha for 'custom' mode (0.0-1.0)")

    # Video parameters
    parser.add_argument("--template_n_frames", type=int, default=-1,
                       help="Number of frames to use from source (-1 = all)")

    # Quality parameters
    parser.add_argument("--sampling_timesteps", type=int, default=50,
                       help="Diffusion sampling steps (25-100)")
    parser.add_argument("--overlap_v2", type=int, default=10,
                       help="Overlap frames for smooth transitions")
    parser.add_argument("--smo_k_d", type=int, default=3,
                       help="Motion smoothing kernel size")
    parser.add_argument("--smo_k_s", type=int, default=13,
                       help="Source video smoothing")
    parser.add_argument("--max_size", type=int, default=1920,
                       help="Maximum resolution")

    # Expression
    parser.add_argument("--emo", type=int, default=4,
                       help="Emotion value (0-7)")
    parser.add_argument("--fade_in", type=int, default=-1,
                       help="Fade in duration in frames")
    parser.add_argument("--fade_out", type=int, default=-1,
                       help="Fade out duration in frames")

    # Other
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")

    args = parser.parse_args()

    # Validate custom mode
    if args.blend_mode == 'custom' and args.blend_alpha is None:
        parser.error("--blend_alpha is required when --blend_mode is 'custom'")

    # Set random seed
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        seed_everything(args.seed)

    # Initialize SDK
    print("Initializing Ditto SDK...")
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    # Prepare configuration
    setup_kwargs = {
        "template_n_frames": args.template_n_frames,
        "smo_k_s": args.smo_k_s,
        "sampling_timesteps": args.sampling_timesteps,
        "overlap_v2": args.overlap_v2,
        "smo_k_d": args.smo_k_d,
        "max_size": args.max_size,
        "emo": args.emo,
    }

    run_kwargs = {
        "fade_in": args.fade_in,
        "fade_out": args.fade_out,
    }

    more_kwargs = {
        "setup_kwargs": setup_kwargs,
        "run_kwargs": run_kwargs,
    }

    # Run inference
    run_with_blend_control(
        SDK=SDK,
        audio_path=args.audio,
        source_path=args.source_video,
        output_path=args.output,
        blend_mode=args.blend_mode,
        blend_alpha=args.blend_alpha,
        more_kwargs=more_kwargs
    )


if __name__ == "__main__":
    main()
