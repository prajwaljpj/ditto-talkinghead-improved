#!/usr/bin/env python3
"""
WebRTC Test Client for Ditto Signaling Server

This client:
1. Connects to signaling server via WebSocket
2. Establishes WebRTC connection
3. Sends audio from a WAV file
4. Receives video and audio streams
5. Saves them to files for sync analysis
6. Measures timestamp differences
"""

import asyncio
import json
import logging
import argparse
import time
from pathlib import Path
import numpy as np
import av
import websockets
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaBlackhole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFileTrack(MediaStreamTrack):
    """
    Audio track that sends audio from a WAV file.
    """
    kind = "audio"

    def __init__(self, audio_file: str, loop: bool = False):
        super().__init__()
        self.audio_file = audio_file
        self.loop = loop

        # Load audio using librosa
        import librosa
        self.audio_data, self.sample_rate = librosa.load(audio_file, sr=48000, mono=True)

        logger.info(f"Loaded audio: {len(self.audio_data)} samples at {self.sample_rate}Hz ({len(self.audio_data)/self.sample_rate:.2f}s)")

        self._timestamp = 0
        self._samples_per_frame = 1920  # 40ms at 48kHz
        self._current_pos = 0
        self._start_time = None

    async def recv(self):
        """Generate audio frames from file."""
        if self._start_time is None:
            self._start_time = time.time()
            logger.info(f"🎤 Started sending audio at t=0.000s")

        # Get next chunk of audio
        start = self._current_pos
        end = start + self._samples_per_frame

        if end > len(self.audio_data):
            if self.loop:
                # Loop back to beginning
                self._current_pos = 0
                start = 0
                end = self._samples_per_frame
            else:
                # End of file - send silence
                audio_chunk = np.zeros(self._samples_per_frame, dtype=np.float32)
        else:
            audio_chunk = self.audio_data[start:end]
            self._current_pos = end

        # Convert to av.AudioFrame
        frame = av.AudioFrame(format='s16', layout='mono', samples=len(audio_chunk))
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = av.Rational(1, self.sample_rate)

        # Convert float32 to int16
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        frame.planes[0].update(audio_int16.tobytes())

        self._timestamp += len(audio_chunk)

        # Log progress
        elapsed = time.time() - self._start_time
        if int(elapsed * 10) % 10 == 0:  # Every second
            progress = self._current_pos / len(self.audio_data) * 100
            logger.info(f"📤 Sending audio: {elapsed:.1f}s, progress: {progress:.1f}%")

        # Pace sending to real-time (40ms between frames)
        await asyncio.sleep(0.04)

        return frame


class SyncAnalyzer:
    """
    Analyzes received video and audio streams for synchronization.
    """

    def __init__(self):
        self.video_frames = []
        self.audio_frames = []
        self.start_time = None

    def on_video_frame(self, frame):
        """Called when video frame is received."""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        self.video_frames.append({
            'wall_time': current_time - self.start_time,
            'pts': frame.pts,
            'time_base': float(frame.time_base),
            'presentation_time': frame.pts * float(frame.time_base) if frame.pts else 0
        })

        if len(self.video_frames) % 25 == 0:
            logger.info(f"📹 Received {len(self.video_frames)} video frames, PTS={frame.pts}, presentation_time={self.video_frames[-1]['presentation_time']:.3f}s")

    def on_audio_frame(self, frame):
        """Called when audio frame is received."""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        self.audio_frames.append({
            'wall_time': current_time - self.start_time,
            'pts': frame.pts,
            'time_base': float(frame.time_base),
            'presentation_time': frame.pts * float(frame.time_base) if frame.pts else 0
        })

        if len(self.audio_frames) % 100 == 0:
            logger.info(f"🔊 Received {len(self.audio_frames)} audio frames, PTS={frame.pts}, presentation_time={self.audio_frames[-1]['presentation_time']:.3f}s")

    def print_analysis(self):
        """Print synchronization analysis."""
        print("\n" + "="*80)
        print("AUDIO-VIDEO SYNCHRONIZATION ANALYSIS")
        print("="*80)

        if len(self.video_frames) == 0 or len(self.audio_frames) == 0:
            print("⚠️ No frames received!")
            return

        print(f"\n📊 RECEIVED FRAMES:")
        print(f"   Video frames: {len(self.video_frames)}")
        print(f"   Audio frames: {len(self.audio_frames)}")

        # Analyze presentation timestamps
        video_pts = [f['presentation_time'] for f in self.video_frames if f['presentation_time'] > 0]
        audio_pts = [f['presentation_time'] for f in self.audio_frames if f['presentation_time'] > 0]

        if len(video_pts) > 0 and len(audio_pts) > 0:
            print(f"\n⏱️ PRESENTATION TIMESTAMPS:")
            print(f"   Video PTS range: {min(video_pts):.3f}s - {max(video_pts):.3f}s")
            print(f"   Audio PTS range: {min(audio_pts):.3f}s - {max(audio_pts):.3f}s")

            # Check if timestamps align
            video_start = min(video_pts)
            audio_start = min(audio_pts)
            start_diff = abs(video_start - audio_start)

            video_end = max(video_pts)
            audio_end = max(audio_pts)
            end_diff = abs(video_end - audio_end)

            print(f"\n🎯 SYNCHRONIZATION:")
            print(f"   Start time difference: {start_diff*1000:.1f}ms")
            print(f"   End time difference: {end_diff*1000:.1f}ms")

            if start_diff > 0.1:
                print(f"   ⚠️ WARNING: Large start time difference!")
                print(f"   Video starts at {video_start:.3f}s, Audio starts at {audio_start:.3f}s")

            if end_diff > 0.1:
                print(f"   ⚠️ WARNING: Large end time difference! Drift occurred.")
                print(f"   Video ends at {video_end:.3f}s, Audio ends at {audio_end:.3f}s")

            # Sample sync points (every second)
            print(f"\n📍 SYNC POINTS (every second):")
            print(f"   {'Time':>6s} | {'Video PTS':>10s} | {'Audio PTS':>10s} | {'Diff':>8s}")
            print(f"   {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

            for t in range(0, int(max(video_end, audio_end)) + 1):
                # Find video frame closest to time t
                video_at_t = [v for v in video_pts if abs(v - t) < 0.1]
                audio_at_t = [a for a in audio_pts if abs(a - t) < 0.1]

                if video_at_t and audio_at_t:
                    v_pts = min(video_at_t, key=lambda x: abs(x - t))
                    a_pts = min(audio_at_t, key=lambda x: abs(x - t))
                    diff = (v_pts - a_pts) * 1000  # ms
                    print(f"   {t:>6.1f}s | {v_pts:>10.3f}s | {a_pts:>10.3f}s | {diff:>7.1f}ms")

        # Analyze wall clock timing
        video_wall = [f['wall_time'] for f in self.video_frames]
        audio_wall = [f['wall_time'] for f in self.audio_frames]

        if len(video_wall) > 0 and len(audio_wall) > 0:
            print(f"\n🕐 WALL CLOCK TIMING:")
            print(f"   Video received over: {max(video_wall):.2f}s")
            print(f"   Audio received over: {max(audio_wall):.2f}s")

            # Check reception rate
            video_fps = len(self.video_frames) / max(video_wall) if max(video_wall) > 0 else 0
            audio_rate = len(self.audio_frames) / max(audio_wall) if max(audio_wall) > 0 else 0

            print(f"   Video reception rate: {video_fps:.1f} FPS (expected: 25 FPS)")
            print(f"   Audio reception rate: {audio_rate:.1f} frames/s (expected: 25 frames/s)")

        print("\n" + "="*80)


class WebRTCTestClient:
    """
    Test client for Ditto WebRTC signaling server.
    """

    def __init__(self, server_url: str, avatar_path: str, audio_path: str, output_dir: str):
        self.server_url = server_url
        self.avatar_path = avatar_path
        self.audio_path = audio_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.ws = None
        self.pc = None
        self.analyzer = SyncAnalyzer()

    async def connect(self):
        """Connect to signaling server via WebSocket."""
        logger.info(f"Connecting to {self.server_url}...")
        self.ws = await websockets.connect(self.server_url)
        logger.info("✓ Connected to signaling server")

    async def send_message(self, message: dict):
        """Send JSON message to server."""
        await self.ws.send(json.dumps(message))

    async def receive_message(self):
        """Receive JSON message from server."""
        message = await self.ws.recv()
        return json.loads(message)

    async def setup_webrtc(self):
        """Setup WebRTC peer connection."""
        # Create peer connection
        configuration = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        self.pc = RTCPeerConnection(configuration=configuration)

        # Store audio track for later
        self.audio_track = AudioFileTrack(self.audio_path, loop=False)

        # Handle incoming tracks
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"📥 Received {track.kind} track")

            if track.kind == "video":
                # Receive video and analyze
                logger.info("Starting video reception and analysis...")
                frame_count = 0
                try:
                    while True:
                        frame = await track.recv()
                        self.analyzer.on_video_frame(frame)
                        frame_count += 1

                        # Stop after reasonable duration
                        if frame_count >= 250:  # ~10 seconds at 25 FPS
                            logger.info(f"Received {frame_count} video frames, stopping...")
                            break
                except Exception as e:
                    logger.error(f"Video reception ended: {e}")

            elif track.kind == "audio":
                # Receive audio and analyze
                logger.info("Starting audio reception and analysis...")
                frame_count = 0
                try:
                    while True:
                        frame = await track.recv()
                        self.analyzer.on_audio_frame(frame)
                        frame_count += 1

                        # Stop after reasonable duration
                        if frame_count >= 250:  # ~10 seconds at 25 FPS
                            logger.info(f"Received {frame_count} audio frames, stopping...")
                            break
                except Exception as e:
                    logger.error(f"Audio reception ended: {e}")

        # Add audio track to send to server
        # This will be in the offer
        self.pc.addTrack(self.audio_track)

        logger.info("✓ WebRTC setup complete")

    async def run(self):
        """Run the test client."""
        try:
            # Connect to signaling server
            await self.connect()

            # Send connect message FIRST (before creating offer)
            logger.info(f"Sending connect message with avatar: {self.avatar_path}")
            await self.send_message({
                "type": "connect",
                "source": self.avatar_path
            })

            # Wait for ready
            msg = await self.receive_message()
            if msg.get("type") == "ready":
                logger.info("✓ Server ready")
            elif msg.get("type") == "error":
                logger.error(f"Server error: {msg.get('message')}")
                return

            # NOW setup WebRTC (after server is ready)
            await self.setup_webrtc()

            # Create offer
            logger.info("Creating WebRTC offer...")
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            # Send offer to server
            await self.send_message({
                "type": "offer",
                "sdp": self.pc.localDescription.sdp
            })

            # Wait for answer
            logger.info("Waiting for answer...")
            msg = await self.receive_message()
            if msg.get("type") == "answer":
                answer = RTCSessionDescription(sdp=msg["sdp"], type="answer")
                await self.pc.setRemoteDescription(answer)
                logger.info("✓ Received answer, WebRTC connection established")
            elif msg.get("type") == "error":
                logger.error(f"Server error: {msg.get('message')}")
                return

            # Handle ICE candidates
            async def handle_messages():
                try:
                    while True:
                        msg = await self.receive_message()
                        if msg.get("type") == "ice-candidate":
                            # Ignore for now, peer reflexive discovery works
                            pass
                        elif msg.get("type") == "error":
                            logger.error(f"Server error: {msg.get('message')}")
                except Exception as e:
                    logger.info(f"Message handling ended: {e}")

            # Start message handling
            asyncio.create_task(handle_messages())

            # Wait for test to complete
            logger.info("Waiting for test to complete (will receive ~10 seconds of video/audio)...")
            await asyncio.sleep(15)  # Wait for reception to complete

            # Print analysis
            self.analyzer.print_analysis()

        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)
        finally:
            # Cleanup
            if self.pc:
                await self.pc.close()
            if self.ws:
                await self.ws.close()
            logger.info("✓ Test client closed")


async def main():
    parser = argparse.ArgumentParser(description="WebRTC Test Client for Ditto")
    parser.add_argument("--server", default="ws://localhost:8080", help="WebSocket server URL")
    parser.add_argument("--avatar", required=True, help="Path to avatar image on server")
    parser.add_argument("--audio", required=True, help="Path to local audio file (WAV)")
    parser.add_argument("--output", default="test_output", help="Output directory for analysis")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("WEBRTC TEST CLIENT")
    logger.info("="*80)
    logger.info(f"Server: {args.server}")
    logger.info(f"Avatar: {args.avatar}")
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Output: {args.output}")
    logger.info("="*80)

    client = WebRTCTestClient(args.server, args.avatar, args.audio, args.output)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
