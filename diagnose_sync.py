#!/usr/bin/env python3
"""
Simple Sync Diagnostic Tool

Analyzes the signaling server logs to detect audio-video desync issues.
"""

import re
import sys


def analyze_logs(log_file=None):
    """Analyze logs from stdin or file."""

    video_frames = []
    audio_chunks = []
    pacing_waits = []

    lines = []
    if log_file:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    else:
        print("Reading from stdin (paste logs or pipe from server)...")
        print("Press Ctrl+D when done.\n")
        lines = sys.stdin.readlines()

    for line in lines:
        line = line.strip()

        # Video frame sent
        if "📤 Frame" in line:
            match = re.search(r'Frame (\d+): audio_ts=([\d.]+)s, audio_clock=([\d.]+)s', line)
            if match:
                frame_idx = int(match.group(1))
                audio_ts = float(match.group(2))
                audio_clock = float(match.group(3))
                video_frames.append({
                    'idx': frame_idx,
                    'audio_ts': audio_ts,
                    'audio_clock': audio_clock,
                    'diff': audio_clock - audio_ts
                })

        # Pacing wait
        if "⏳ PACING:" in line:
            match = re.search(r'Waiting ([\d.]+)ms', line)
            if match:
                wait_ms = float(match.group(1))
                pacing_waits.append(wait_ms)

        # Status updates
        if "📊 STATUS:" in line:
            match = re.search(r'(\d+) frames in ([\d.]+)s = ([\d.]+) FPS', line)
            if match:
                frames = int(match.group(1))
                duration = float(match.group(2))
                fps = float(match.group(3))
                print(f"Status: {frames} frames, {duration:.1f}s, {fps:.1f} FPS")

    print("\n" + "="*80)
    print("SYNCHRONIZATION DIAGNOSTIC")
    print("="*80)

    if len(video_frames) == 0:
        print("⚠️ No video frame data found in logs!")
        print("Make sure server is running with logging enabled.")
        return

    print(f"\n📊 FRAMES ANALYZED: {len(video_frames)}")

    # Analyze audio_ts vs audio_clock difference
    diffs = [f['diff'] for f in video_frames]
    avg_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    min_diff = min(diffs)

    print(f"\n🎯 AUDIO TIMESTAMP vs AUDIO CLOCK:")
    print(f"   Average difference: {avg_diff*1000:.1f}ms")
    print(f"   Min difference: {min_diff*1000:.1f}ms")
    print(f"   Max difference: {max_diff*1000:.1f}ms")

    if avg_diff > 0.1:
        print(f"\n   ⚠️ PROBLEM DETECTED: Audio clock is ahead of audio timestamp by {avg_diff*1000:.1f}ms")
        print(f"   This means: Audio is playing BEFORE the video frame it belongs to!")
        print(f"   Result: You hear the audio before seeing the corresponding lip movement")
        print(f"\n   ROOT CAUSE: Audio is added to playback buffer immediately,")
        print(f"   but video frames are paced/delayed, causing audio to run ahead.")
    elif avg_diff < -0.1:
        print(f"\n   ⚠️ PROBLEM DETECTED: Audio timestamp is ahead of audio clock by {-avg_diff*1000:.1f}ms")
        print(f"   This means: Video is playing BEFORE the audio it belongs to!")
        print(f"   Result: You see lip movement before hearing the sound")
    else:
        print(f"\n   ✓ Audio and video timestamps are aligned (within 100ms)")

    # Check if difference is growing (drift)
    if len(video_frames) >= 10:
        early_diffs = diffs[:5]
        late_diffs = diffs[-5:]
        early_avg = sum(early_diffs) / len(early_diffs)
        late_avg = sum(late_diffs) / len(late_diffs)
        drift = late_avg - early_avg

        print(f"\n📈 DRIFT ANALYSIS:")
        print(f"   Early frames (first 5): {early_avg*1000:.1f}ms difference")
        print(f"   Late frames (last 5): {late_avg*1000:.1f}ms difference")
        print(f"   Drift: {drift*1000:.1f}ms")

        if abs(drift) > 0.05:
            print(f"   ⚠️ DRIFT DETECTED: Sync is getting worse over time!")
        else:
            print(f"   ✓ No significant drift detected")

    # Analyze pacing
    if len(pacing_waits) > 0:
        avg_wait = sum(pacing_waits) / len(pacing_waits)
        print(f"\n⏱️ FRAME PACING:")
        print(f"   Pacing waits: {len(pacing_waits)} times")
        print(f"   Average wait: {avg_wait:.1f}ms")
        print(f"   Expected wait: ~20ms (model generates 50 FPS, we output 25 FPS)")

        if avg_wait > 30:
            print(f"   ⚠️ Long waits suggest video is being delayed significantly")

    # Show timeline
    print(f"\n⏰ SYNC TIMELINE (showing every 25th frame):")
    print(f"   {'Frame':>6s} | {'Audio TS':>10s} | {'Audio Clock':>12s} | {'Diff':>8s} | {'Status'}")
    print(f"   {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*20}")

    for i, f in enumerate(video_frames):
        if i % 25 == 0 or i < 5:  # Show first 5 and every 25th
            diff_ms = f['diff'] * 1000
            status = "OK" if abs(diff_ms) < 100 else "DESYNC!"
            print(f"   {f['idx']:>6d} | {f['audio_ts']:>10.3f}s | {f['audio_clock']:>12.3f}s | {diff_ms:>7.1f}ms | {status}")

    print("\n" + "="*80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if avg_diff > 0.1:
        print("   🔧 FIX: Audio is playing ahead of video.")
        print("   The issue is that audio is added to buffer immediately when video")
        print("   frame is ready, but then video frame is paced (delayed).")
        print("\n   Solution: Remove the pacing delay from video, let WebRTC handle")
        print("   timing based on PTS timestamps. Or delay audio addition by same amount.")
    elif avg_diff < -0.1:
        print("   🔧 FIX: Video is playing ahead of audio.")
        print("   The video frames are being sent before their audio is ready.")
    else:
        print("   ✓ Timestamps look good! If you still see desync, the issue may be:")
        print("   - WebRTC buffering/jitter buffer settings")
        print("   - Network latency variations")
        print("   - Browser playback timing")

    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose audio-video sync from logs")
    parser.add_argument("--file", help="Log file to analyze (default: read from stdin)")
    args = parser.parse_args()

    analyze_logs(args.file)
