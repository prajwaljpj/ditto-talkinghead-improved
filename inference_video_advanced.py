"""
Advanced Video Inference with BOTH Motion Preservation AND Blend Control

This script combines:
1. Motion Preservation: Generate lip-sync from neutral frames, composite to ALL frames
2. Blend Control: Control how much rendered vs original face is used

Perfect for looping videos with natural movements and expressions you want to preserve
while having full control over the blending ratio.
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


def run_advanced_inference(
    SDK: StreamSDK,
    audio_path: str,
    source_video_path: str,
    output_path: str,
    motion_template_frames: int = 100,
    blend_mode: str = 'blend',
    blend_alpha: float = None,
    more_kwargs: dict = None
):
    """
    Run inference with BOTH motion preservation AND blend control.

    Args:
        SDK: Initialized StreamSDK
        audio_path: Path to audio file
        source_video_path: Path to source video
        output_path: Path to output video
        motion_template_frames: Number of frames for motion generation (neutral part)
        blend_mode: 'blend', 'strong', 'replace', 'weak', or 'custom'
        blend_alpha: Alpha for 'custom' mode (0.0-1.0)
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
    print("ADVANCED VIDEO INFERENCE")
    print("=" * 70)
    print(f"Source video: {source_video_path}")
    print(f"Audio file: {audio_path}")
    print(f"Output path: {output_path}")
    print(f"\nFeature 1: Motion Preservation")
    print(f"  - Use first {motion_template_frames} frames for MOTION generation")
    print(f"  - Use ALL frames for COMPOSITING (preserves natural movements)")
    print(f"\nFeature 2: Blend Control")
    print(f"  - Blend mode: {blend_mode}")
    if blend_mode == 'custom':
        print(f"  - Custom alpha: {blend_alpha} (0.0=original, 1.0=rendered)")
    print("=" * 70)

    # Step 1: Replace PutBack with configurable version for blend control
    from core.atomic_components.putback_configurable import get_putback_by_mode
    SDK.putback = get_putback_by_mode(
        blend_mode=blend_mode,
        custom_alpha=blend_alpha,
    )
    print("\n[1/7] Blend control configured")

    # Step 2: Load FULL video for compositing
    print(f"[2/7] Loading full source video for compositing...")
    full_rgb_list, is_image_flag = load_source_frames(
        source_video_path,
        max_dim=setup_kwargs.get("max_size", 1920),
        n_frames=-1  # Load ALL frames
    )
    total_source_frames = len(full_rgb_list)
    print(f"      Loaded {total_source_frames} frames from source video")

    # Step 3: Setup SDK with limited frames for motion generation
    print(f"[3/7] Setting up motion generation (using first {motion_template_frames} frames)...")
    setup_kwargs["template_n_frames"] = motion_template_frames
    SDK.setup(source_video_path, output_path, **setup_kwargs)
    print(f"      Motion templates: {SDK.source_info_frames} frames")

    # Step 4: Replace compositing frames with FULL video
    print(f"[4/7] Extracting compositing info from all {total_source_frames} frames...")

    # Save limited motion features
    motion_x_s_info_lst = SDK.source_info["x_s_info_lst"].copy()
    motion_f_s_lst = SDK.source_info["f_s_lst"].copy()

    # Extract features from ALL frames for compositing
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

    for i, rgb in enumerate(full_rgb_list):
        if i % 50 == 0:
            print(f"      Processing frame {i}/{total_source_frames}...")
        info = SDK.avatar_registrar.source2info(rgb, last_lmk, **crop_kwargs)
        for k in keys:
            full_source_info[f"{k}_lst"].append(info[k])
        last_lmk = info["lmk203"]

    # Replace compositing data
    SDK.source_info["x_s_info_lst"] = full_source_info["x_s_info_lst"]
    SDK.source_info["f_s_lst"] = full_source_info["f_s_lst"]
    SDK.source_info["M_c2o_lst"] = full_source_info["M_c2o_lst"]
    SDK.source_info["img_rgb_lst"] = full_rgb_list
    SDK.source_info_frames = len(full_source_info["x_s_info_lst"])

    print(f"      ✓ Motion generation: {len(motion_x_s_info_lst)} frames (neutral)")
    print(f"      ✓ Compositing: {SDK.source_info_frames} frames (all)")
    print(f"      ✓ Blend mode: {blend_mode}")

    # Step 5: Load and process audio
    print(f"[5/7] Processing audio...")
    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    print(f"      Audio duration: {len(audio) / 16000:.2f} seconds")
    print(f"      Output frames: {num_f} frames")

    # Setup frame count
    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    # Step 6: Generate motion and video
    print(f"[6/7] Generating lip-sync motion and rendering...")
    online_mode = SDK.online_mode
    if online_mode:
        print("      Using ONLINE mode...")
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        print("      Using OFFLINE mode...")
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)

    print("      Rendering frames...")
    SDK.close()

    # Step 7: Add audio
    print(f"[7/7] Muxing audio with video...")
    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    os.system(cmd)

    print("\n" + "=" * 70)
    print("✓ Advanced video generation complete!")
    print(f"✓ Output: {output_path}")
    print("\nWhat was preserved from original video:")
    print("  ✓ Natural head movements (from all frames)")
    print("  ✓ Eye blinks (from all frames)")
    print("  ✓ Body movements (from all frames)")
    print("  ✓ Facial expressions like smiles (from all frames)")
    print("\nWhat was generated:")
    print("  ✓ Lip movements matching audio (from neutral frames)")
    print("  ✓ Speech-related facial expressions")
    print(f"\nBlending: {blend_mode} mode")
    if blend_mode == 'strong':
        print("  → 90% rendered, 10% original (smile/teeth hidden)")
    elif blend_mode == 'blend':
        print("  → 50% rendered, 50% original (balanced)")
    elif blend_mode == 'replace':
        print("  → 100% rendered, 0% original (full replacement)")
    elif blend_mode == 'weak':
        print("  → 25% rendered, 75% original (subtle)")
    elif blend_mode == 'custom':
        print(f"  → {int(blend_alpha*100)}% rendered, {int((1-blend_alpha)*100)}% original")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced video inference with motion preservation AND blend control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script combines TWO powerful features:

1. MOTION PRESERVATION:
   - Uses first N frames for motion generation (neutral part)
   - Uses ALL frames for compositing (preserves natural movements)
   - Result: Lip-sync from neutral + blinks/head movements from all frames

2. BLEND CONTROL:
   - Controls how much rendered vs original face is used
   - 5 modes: blend, strong, replace, weak, custom
   - Result: Fine-tune output quality and hide problematic expressions

Examples:

  1. Looping video with smile (preserve movements, hide smile):
    python inference_video_advanced.py \\
        --source_video looping_avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 100 \\
        --blend_mode strong

  2. Natural video with blinks (preserve, balanced blend):
    python inference_video_advanced.py \\
        --source_video natural_avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 100 \\
        --blend_mode blend

  3. Perfect control (custom percentage):
    python inference_video_advanced.py \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 100 \\
        --blend_mode custom \\
        --blend_alpha 0.7

  4. Maximum quality with all features:
    python inference_video_advanced.py \\
        --data_root ./checkpoints/ditto_trt_Ampere_Plus \\
        --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \\
        --source_video avatar.mp4 \\
        --audio speech.wav \\
        --output result.mp4 \\
        --motion_frames 100 \\
        --blend_mode strong \\
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

    # Motion preservation
    parser.add_argument("--motion_frames", type=int, default=100,
                       help="Number of frames for motion generation (neutral part)")

    # Blend control
    parser.add_argument("--blend_mode", type=str, default='blend',
                       choices=['blend', 'strong', 'replace', 'weak', 'custom'],
                       help="Blending mode (default: blend)")
    parser.add_argument("--blend_alpha", type=float, default=None,
                       help="Custom alpha for 'custom' mode (0.0-1.0)")

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

    # Run advanced inference
    run_advanced_inference(
        SDK=SDK,
        audio_path=args.audio,
        source_video_path=args.source_video,
        output_path=args.output,
        motion_template_frames=args.motion_frames,
        blend_mode=args.blend_mode,
        blend_alpha=args.blend_alpha,
        more_kwargs=more_kwargs
    )


if __name__ == "__main__":
    main()
