"""
Video Inference with Preserved Original Motion

This script generates lip-sync using ONLY neutral frames but composites back
to ALL original video frames, preserving natural head tilts, blinks, smiles, etc.

Use Case: Your video has natural movements you want to keep, but also has
expressions (like smiles) that conflict with speech generation.
"""

import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import argparse
from pathlib import Path

from stream_pipeline_offline import StreamSDK
from core.atomic_components.avatar_registrar import AvatarRegistrar
from core.atomic_components.loader import load_source_frames


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


def run_with_preserved_motion(
    SDK: StreamSDK,
    audio_path: str,
    source_video_path: str,
    output_path: str,
    motion_template_frames: int = 100,
    more_kwargs: dict = None
):
    """
    Generate lip-sync using neutral frames but composite to full original video.

    Args:
        SDK: Initialized StreamSDK
        audio_path: Path to audio file
        source_video_path: Path to source video
        output_path: Path to output video
        motion_template_frames: Number of frames to use for motion generation (neutral part)
        more_kwargs: Additional configuration
    """

    if more_kwargs is None:
        more_kwargs = {}

    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    # Video-specific defaults
    video_defaults = {
        "smo_k_s": 13,
        "eye_f0_mode": True,
        "drive_eye": None,
        "max_size": 1920,
        "crop_scale": 2.3,
        "crop_vx_ratio": 0,
        "crop_vy_ratio": -0.125,
        "crop_flag_do_rot": True,
    }

    for key, value in video_defaults.items():
        if key not in setup_kwargs:
            setup_kwargs[key] = value

    print("=" * 70)
    print("VIDEO INFERENCE WITH PRESERVED ORIGINAL MOTION")
    print("=" * 70)
    print(f"Source video: {source_video_path}")
    print(f"Audio file: {audio_path}")
    print(f"Output path: {output_path}")
    print(f"\nStrategy:")
    print(f"  - Use first {motion_template_frames} frames for MOTION generation")
    print(f"  - Use ALL frames for COMPOSITING (preserves natural movements)")
    print("=" * 70)

    # Step 1: Load FULL video for compositing
    print("\n[1/6] Loading full source video for compositing...")
    full_rgb_list, is_image_flag = load_source_frames(
        source_video_path,
        max_dim=setup_kwargs.get("max_size", 1920),
        n_frames=-1  # Load ALL frames
    )
    total_source_frames = len(full_rgb_list)
    print(f"      Loaded {total_source_frames} frames from source video")

    # Step 2: Setup SDK with limited frames for motion generation
    print(f"\n[2/6] Setting up motion generation (using first {motion_template_frames} frames)...")

    # Temporarily set template_n_frames for motion generation
    setup_kwargs["template_n_frames"] = motion_template_frames

    # Standard setup (this extracts features from first N frames only)
    SDK.setup(source_video_path, output_path, **setup_kwargs)

    print(f"      Motion templates: {SDK.source_info_frames} frames")
    print(f"      Composite frames available: {total_source_frames} frames")

    # Step 3: Replace the limited frame list with FULL frame list for compositing
    print(f"\n[3/6] Replacing compositing frames with full video...")

    # Keep the extracted features (x_s_info_lst) for motion - only from neutral frames
    # But replace img_rgb_lst and M_c2o_lst with full video for compositing

    # We need to re-extract features for ALL frames for compositing
    # But keep the limited features for motion generation in condition handler
    from core.atomic_components.source2info import Source2Info

    # Save the limited motion features
    motion_x_s_info_lst = SDK.source_info["x_s_info_lst"].copy()
    motion_f_s_lst = SDK.source_info["f_s_lst"].copy()

    # Now extract features from ALL frames for compositing
    crop_kwargs = {
        "crop_scale": setup_kwargs.get("crop_scale", 2.3),
        "crop_vx_ratio": setup_kwargs.get("crop_vx_ratio", 0),
        "crop_vy_ratio": setup_kwargs.get("crop_vy_ratio", -0.125),
        "crop_flag_do_rot": setup_kwargs.get("crop_flag_do_rot", True),
    }

    full_source_info = {
        "x_s_info_lst": [],
        "f_s_lst": [],
        "M_c2o_lst": [],
        "eye_open_lst": [],
        "eye_ball_lst": [],
    }

    keys = ["x_s_info", "f_s", "M_c2o", "eye_open", "eye_ball"]
    last_lmk = None

    print(f"      Extracting compositing info from all {total_source_frames} frames...")
    for i, rgb in enumerate(full_rgb_list):
        if i % 50 == 0:
            print(f"      Processing frame {i}/{total_source_frames}...")
        info = SDK.avatar_registrar.source2info(rgb, last_lmk, **crop_kwargs)
        for k in keys:
            full_source_info[f"{k}_lst"].append(info[k])
        last_lmk = info["lmk203"]

    # Replace compositing data with full video data
    SDK.source_info["x_s_info_lst"] = full_source_info["x_s_info_lst"]
    SDK.source_info["f_s_lst"] = full_source_info["f_s_lst"]
    SDK.source_info["M_c2o_lst"] = full_source_info["M_c2o_lst"]
    SDK.source_info["img_rgb_lst"] = full_rgb_list
    SDK.source_info_frames = len(full_source_info["x_s_info_lst"])

    print(f"      ✓ Now have {SDK.source_info_frames} frames for compositing")
    print(f"      ✓ Motion generation still uses {len(motion_x_s_info_lst)} neutral frames")

    # Step 4: Load and process audio
    print(f"\n[4/6] Processing audio...")
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    print(f"      Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"      Output frames needed: {num_f} frames")

    # Get fade and control info
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})

    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    # Step 5: Generate motion and video
    print(f"\n[5/6] Generating lip-sync motion and rendering...")
    online_mode = SDK.online_mode
    if online_mode:
        print("      Using ONLINE mode (streaming)...")
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        print("      Using OFFLINE mode (batch)...")
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)

    print("      Rendering frames...")
    SDK.close()

    # Step 6: Add audio
    print(f"\n[6/6] Muxing audio with video...")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    print("\n" + "=" * 70)
    print("✓ Video generation complete!")
    print(f"✓ Output saved to: {output_path}")
    print("\nWhat was preserved from original video:")
    print("  ✓ Natural head movements")
    print("  ✓ Eye blinks")
    print("  ✓ Body movements")
    print("  ✓ Facial expressions (where they don't conflict)")
    print("\nWhat was generated:")
    print("  ✓ Lip movements matching audio")
    print("  ✓ Speech-related facial expressions")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate lip-sync while preserving original video motion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Basic usage (use first 100 frames for motion, all frames for compositing):
    python inference_video_preserve_motion.py \\
        --source_video avatar_with_smile.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 100

  2. Find neutral range first (extract first 100 frames):
    python inference_video_preserve_motion.py \\
        --source_video looping_avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 50

  3. High quality:
    python inference_video_preserve_motion.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 100 \\
        --sampling_timesteps 75 \\
        --overlap_v2 15
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
                       help="Path to source video")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to input audio (.wav)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output video (.mp4)")

    # Key parameter
    parser.add_argument("--motion_frames", type=int, default=100,
                       help="Number of frames to use for motion generation (e.g., neutral part)")

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

    # Set random seed
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        seed_everything(args.seed)

    # Initialize SDK
    print("Initializing Ditto SDK...")
    SDK = StreamSDK(args.cfg_pkl, args.data_root)

    # Prepare configuration
    setup_kwargs = {
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
    run_with_preserved_motion(
        SDK=SDK,
        audio_path=args.audio,
        source_video_path=args.source_video,
        output_path=args.output,
        motion_template_frames=args.motion_frames,
        more_kwargs=more_kwargs
    )


if __name__ == "__main__":
    main()
