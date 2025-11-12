#!/usr/bin/env python3
"""
Real-time synchronization monitor for debugging WebRTC streaming

Analyzes the log output from signaling_server.py to identify sync issues.
"""

import re
import sys
from collections import deque
from datetime import datetime


class SyncMonitor:
    def __init__(self):
        self.audio_chunks_stored = []
        self.audio_chunks_retrieved = []
        self.frames_sent = []
        self.frames_dropped = []
        self.frames_waiting = []
        self.audio_clock_updates = []
        self.start_time = None

    def parse_line(self, line: str):
        """Parse a log line and extract sync information"""

        # Extract timestamp if present
        timestamp_match = re.search(r't=(\d+\.\d+)', line)
        if timestamp_match:
            t = float(timestamp_match.group(1))
            if self.start_time is None:
                self.start_time = t

        # Audio chunk stored
        if "📥 STORE:" in line:
            match = re.search(r'queue_idx=(\d+), timestamp=([\d.]+)s, 48kHz samples=(\d+), RMS=([\d.]+)', line)
            if match:
                idx = int(match.group(1))
                ts = float(match.group(2))
                samples = int(match.group(3))
                rms = float(match.group(4))
                self.audio_chunks_stored.append({
                    'idx': idx,
                    'timestamp': ts,
                    'samples': samples,
                    'rms': rms
                })

        # Audio chunk retrieved
        if "📤 RETRIEVE:" in line:
            match = re.search(r'frame=(\d+), audio_ts=([\d.]+)s, audio_clock=([\d.]+)s, queue_remaining=(\d+)', line)
            if match:
                frame_idx = int(match.group(1))
                audio_ts = float(match.group(2))
                audio_clock = float(match.group(3))
                queue_remaining = int(match.group(4))
                self.audio_chunks_retrieved.append({
                    'frame_idx': frame_idx,
                    'audio_ts': audio_ts,
                    'audio_clock': audio_clock,
                    'queue_remaining': queue_remaining
                })

        # Frame waiting
        if "⏳ WAITING:" in line:
            match = re.search(r'frame (\d+) at ([\d.]+)s, audio_clock=([\d.]+)s, waiting ([\d.]+)ms', line)
            if match:
                frame_idx = int(match.group(1))
                frame_ts = float(match.group(2))
                audio_clock = float(match.group(3))
                wait_ms = float(match.group(4))
                self.frames_waiting.append({
                    'frame_idx': frame_idx,
                    'frame_ts': frame_ts,
                    'audio_clock': audio_clock,
                    'wait_ms': wait_ms
                })

        # Frame dropped
        if "⚠️ DROP FRAME" in line:
            match = re.search(r'DROP FRAME (\d+): too late by ([\d.]+)ms \(audio_clock=([\d.]+)s, frame_time=([\d.]+)s\)', line)
            if match:
                frame_idx = int(match.group(1))
                late_ms = float(match.group(2))
                audio_clock = float(match.group(3))
                frame_time = float(match.group(4))
                self.frames_dropped.append({
                    'frame_idx': frame_idx,
                    'late_ms': late_ms,
                    'audio_clock': audio_clock,
                    'frame_time': frame_time
                })
                print(f"⚠️ DROPPED FRAME {frame_idx}: late by {late_ms:.1f}ms (audio_clock={audio_clock:.3f}s, frame_time={frame_time:.3f}s)")

        # Frame synced
        if "✓ SYNCED:" in line:
            match = re.search(r'frame (\d+) released at audio_clock=([\d.]+)s', line)
            if match:
                frame_idx = int(match.group(1))
                audio_clock = float(match.group(2))
                self.frames_sent.append({
                    'frame_idx': frame_idx,
                    'audio_clock': audio_clock
                })

        # Audio clock update
        if "audio_clock=" in line and "passthrough frames" in line:
            match = re.search(r'Sent (\d+) passthrough frames.*audio_clock=([\d.]+)s', line)
            if match:
                frame_count = int(match.group(1))
                audio_clock = float(match.group(2))
                self.audio_clock_updates.append({
                    'frame_count': frame_count,
                    'audio_clock': audio_clock
                })

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SYNCHRONIZATION MONITOR SUMMARY")
        print("="*80)

        print(f"\n📊 AUDIO CHUNKS:")
        print(f"   Stored: {len(self.audio_chunks_stored)}")
        print(f"   Retrieved: {len(self.audio_chunks_retrieved)}")
        print(f"   Remaining: {len(self.audio_chunks_stored) - len(self.audio_chunks_retrieved)}")

        print(f"\n🎬 VIDEO FRAMES:")
        print(f"   Sent: {len(self.frames_sent)}")
        print(f"   Waiting: {len(self.frames_waiting)}")
        print(f"   Dropped: {len(self.frames_dropped)}")

        if len(self.frames_dropped) > 0:
            print(f"\n⚠️ FRAME DROP ANALYSIS:")
            late_times = [f['late_ms'] for f in self.frames_dropped]
            print(f"   Average lateness: {sum(late_times)/len(late_times):.1f}ms")
            print(f"   Max lateness: {max(late_times):.1f}ms")
            print(f"   Min lateness: {min(late_times):.1f}ms")

            print(f"\n   Drop rate: {len(self.frames_dropped)/(len(self.frames_sent)+len(self.frames_dropped))*100:.1f}%")

            if len(self.frames_dropped) > 10:
                print(f"\n   ⚠️ HIGH DROP RATE! Possible causes:")
                print(f"      1. Model inference is too slow (can't keep up with real-time)")
                print(f"      2. Large delay between audio input and video output")
                print(f"      3. Irregular frame generation rate")

        if len(self.frames_waiting) > 0:
            print(f"\n⏳ WAITING ANALYSIS:")
            wait_times = [f['wait_ms'] for f in self.frames_waiting]
            print(f"   Average wait: {sum(wait_times)/len(wait_times):.1f}ms")
            print(f"   Max wait: {max(wait_times):.1f}ms")
            print(f"   Min wait: {min(wait_times):.1f}ms")

        if len(self.audio_clock_updates) > 1:
            print(f"\n🎵 AUDIO CLOCK:")
            print(f"   Final audio clock: {self.audio_clock_updates[-1]['audio_clock']:.3f}s")

        print("\n" + "="*80)


def main():
    print("Real-time Synchronization Monitor")
    print("Reading from stdin (pipe signaling_server.py output here)")
    print("Example: python signaling_server.py ... 2>&1 | python monitor_sync.py")
    print("="*80 + "\n")

    monitor = SyncMonitor()

    try:
        for line in sys.stdin:
            line = line.strip()
            if line:
                monitor.parse_line(line)
                # Also print the original line
                print(line)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    # Print summary
    monitor.print_summary()


if __name__ == "__main__":
    main()
