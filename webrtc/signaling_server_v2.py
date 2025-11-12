"""
WebRTC Signaling Server for Ditto Avatar - Clean Architecture v2

This is a complete rewrite with clean separation of concerns:
- AudioPassthrough: Handle audio reception and conversion
- VideoGenerator: Interface with Ditto SDK
- AVSynchronizer: Coordinate audio-video timing
- WebRTC Session: Orchestrate components

Design principles:
1. Single source of timestamps (from Ditto model)
2. WebRTC handles all timing via PTS
3. No manual pacing or artificial delays
4. Clear error boundaries and backpressure
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from fractions import Fraction
from collections import deque

import numpy as np
import scipy.signal
import websockets
from websockets.legacy.server import WebSocketServerProtocol
import av

# WebRTC
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.mediastreams import AudioStreamTrack

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ditto components
from stream_pipeline_online import StreamSDK

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPassthrough:
    """
    Handles incoming audio from client:
    - Converts stereo to mono
    - Downsamples to 16kHz for Ditto
    - Stores 48kHz mono for playback with timestamps
    """

    def __init__(self):
        self.input_rate = 48000  # WebRTC/browser standard
        self.model_rate = 16000  # Ditto requirement
        self.output_rate = 48000  # Playback rate

        # Audio buffers
        self.buffer_16k = np.array([], dtype=np.float32)  # For model
        self.buffer_48k = np.array([], dtype=np.float32)  # For playback

        # Timing
        self.total_samples_received = 0  # At 48kHz
        self.chunk_size_16k = 6400  # 400ms @ 16kHz (required by Ditto)
        self.chunk_size_48k = 1920  # 40ms @ 48kHz (standard WebRTC frame)

        # Playback chunks WITHOUT timestamps (timestamp assigned when paired with video)
        self.playback_chunks = deque(maxlen=500)  # ~20 seconds @ 25fps

        # Event for efficient signaling when audio is ready
        self.audio_ready = asyncio.Event()

        self._frame_count = 0

    def add_frame(self, frame: av.AudioFrame) -> bool:
        """
        Add incoming audio frame from client.
        Returns True if a new 16kHz chunk is ready for the model.
        """
        self._frame_count += 1

        # Convert to numpy
        audio_array = frame.to_ndarray()

        # Convert stereo to mono by averaging channels
        audio_mono = self._to_mono(audio_array, frame.layout.name)

        # Normalize to float32 [-1, 1]
        audio_normalized = self._normalize(audio_mono, audio_array.dtype)

        # Log first frame
        if self._frame_count == 1:
            logger.info(f"📥 First audio frame: {frame.sample_rate}Hz, "
                       f"{frame.layout.name}, {len(audio_normalized)} samples")
            logger.info(f"   Range: [{audio_normalized.min():.3f}, {audio_normalized.max():.3f}], "
                       f"RMS: {np.sqrt(np.mean(audio_normalized**2)):.4f}")

        # Track samples received
        self.total_samples_received += len(audio_normalized)

        # Store 48kHz for playback
        self.buffer_48k = np.concatenate([self.buffer_48k, audio_normalized])

        # Downsample to 16kHz for model
        if frame.sample_rate == 48000:
            audio_16k = scipy.signal.resample_poly(audio_normalized, up=1, down=3)
        elif frame.sample_rate == 16000:
            audio_16k = audio_normalized
        else:
            # Generic resampling
            num_samples = int(len(audio_normalized) * 16000 / frame.sample_rate)
            audio_16k = scipy.signal.resample(audio_normalized, num_samples)

        self.buffer_16k = np.concatenate([self.buffer_16k, audio_16k])

        # Check if we have enough for a model chunk
        return len(self.buffer_16k) >= self.chunk_size_16k

    def get_16khz_chunk(self) -> Optional[np.ndarray]:
        """Get next 400ms chunk for Ditto model (6400 samples @ 16kHz)."""
        if len(self.buffer_16k) < self.chunk_size_16k:
            return None

        chunk = self.buffer_16k[:self.chunk_size_16k]
        self.buffer_16k = self.buffer_16k[self.chunk_size_16k:]

        # Also extract corresponding 48kHz audio for playback
        # 6400 samples @ 16kHz = 400ms = 19200 samples @ 48kHz
        samples_48k_needed = self.chunk_size_16k * 3

        if len(self.buffer_48k) >= samples_48k_needed:
            chunk_48k = self.buffer_48k[:samples_48k_needed]
            self.buffer_48k = self.buffer_48k[samples_48k_needed:]

            # Split 400ms chunk into 10 x 40ms sub-chunks (for 25fps)
            # Each sub-chunk = 1920 samples @ 48kHz
            # NOTE: We store audio WITHOUT timestamps here
            # Timestamps will be assigned later when paired with video frames!
            for i in range(0, len(chunk_48k), self.chunk_size_48k):
                sub_chunk = chunk_48k[i:i + self.chunk_size_48k]

                # Pad if needed (last chunk might be shorter)
                if len(sub_chunk) < self.chunk_size_48k:
                    sub_chunk = np.pad(sub_chunk, (0, self.chunk_size_48k - len(sub_chunk)))

                # Store audio without timestamp (timestamp comes from video frame!)
                self.playback_chunks.append(sub_chunk)

            # Signal that audio is now available (wake up waiting video)
            self.audio_ready.set()

            if len(self.playback_chunks) % 25 == 0:
                logger.debug(f"📦 Stored {len(self.playback_chunks)} audio chunks "
                           f"(~{len(self.playback_chunks)/25:.1f}s of audio)")

        return chunk

    def get_next_playback_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk (without timestamp - timestamp comes from video!)"""
        if len(self.playback_chunks) > 0:
            return self.playback_chunks.popleft()
        return None


    def _to_mono(self, audio: np.ndarray, layout: str) -> np.ndarray:
        """Convert stereo/multi-channel to mono by averaging."""
        is_stereo = (layout == 'stereo')

        if audio.ndim > 1:
            # Multi-dimensional array
            if audio.shape[0] == 2:  # (2, N) - channels first
                return audio.mean(axis=0)
            elif audio.shape[1] == 2:  # (N, 2) - samples first
                return audio.mean(axis=1)
            elif is_stereo and audio.shape[0] == 1:
                # Interleaved: (1, N) where N = 2*samples
                audio_flat = audio[0]
                return audio_flat.reshape(-1, 2).mean(axis=1)
            else:
                # Take first channel
                return audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
        elif is_stereo and audio.ndim == 1:
            # 1D interleaved stereo: [L1,R1,L2,R2,...]
            return audio.reshape(-1, 2).mean(axis=1)
        else:
            # Already mono
            return audio

    def _normalize(self, audio: np.ndarray, dtype) -> np.ndarray:
        """Normalize audio to float32 [-1, 1]."""
        if dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            return audio.astype(np.float32) / 2147483648.0
        else:
            # Already float
            audio_float = audio.astype(np.float32)
            max_abs = np.abs(audio_float).max()
            if max_abs > 10.0:  # Definitely not normalized
                return audio_float / max_abs
            return audio_float


class VideoGenerator:
    """
    Interfaces with Ditto SDK to generate video frames.
    Receives frames via callback and stores them with timestamps.

    Note: Frame rate pacing is handled by AVSynchronizer waiting for audio,
    not by dropping frames here.
    """

    def __init__(self, sdk: StreamSDK, target_fps: int = 25):
        self.sdk = sdk
        self.target_fps = target_fps
        # Larger queue to buffer model bursts (model is 2x real-time)
        self.frame_queue = asyncio.Queue(maxsize=200)
        self._frames_generated = 0
        self._frames_dropped = 0  # Track queue full drops
        self._first_frame_time = None

    def on_frame(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """Callback from StreamSDK when frame is ready."""
        self._frames_generated += 1

        try:
            receive_time = time.time()

            # Queue all frames - pacing happens in AVSynchronizer
            self.frame_queue.put_nowait((frame_rgb, frame_idx, timestamp, receive_time))

            if self._first_frame_time is None:
                self._first_frame_time = receive_time
                logger.info(f"🎬 First video frame queued: idx={frame_idx}, "
                           f"timestamp={timestamp:.3f}s")

            # Log generation stats occasionally
            if self._frames_generated % 100 == 0:
                queue_size = self.frame_queue.qsize()
                logger.info(f"📊 Generated {self._frames_generated} frames, "
                           f"queue size: {queue_size}/100")

        except asyncio.QueueFull:
            # CRITICAL: Do NOT drop frames - this breaks FIFO ordering with audio!
            # Instead, log warning (this shouldn't happen with maxsize=200 for 2x real-time model)
            self._frames_dropped += 1
            if self._frames_dropped == 1 or self._frames_dropped % 50 == 0:
                logger.error(f"❌ Video queue full! This breaks audio sync. "
                           f"Dropped {self._frames_dropped} frames. "
                           f"Consider: slower model, faster network, or larger queue.")

    async def get_next_frame(self) -> Tuple[np.ndarray, int, float]:
        """Get next video frame with its timestamp."""
        # Wait indefinitely for next frame (no timeout)
        # The Ditto model will produce frames as fast as it can
        frame_rgb, frame_idx, timestamp, receive_time = await self.frame_queue.get()
        return (frame_rgb, frame_idx, timestamp)

    def feed_audio(self, audio_chunk: np.ndarray):
        """Feed 16kHz audio chunk to Ditto model."""
        if self.sdk:
            self.sdk.run_chunk(audio_chunk, chunksize=(3, 5, 2))


class AVSynchronizer:
    """
    Synchronizes audio and video using timestamps.
    Provides matched (video, audio) pairs for WebRTC.
    """

    def __init__(self, video_generator: VideoGenerator, audio_passthrough: AudioPassthrough):
        self.video_gen = video_generator
        self.audio_pass = audio_passthrough
        self._sync_count = 0

    async def get_next(self) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Get next synchronized frame.

        CRITICAL: Uses simple FIFO pairing - audio chunk N gets video frame N's timestamp.
        This ensures perfect sync since we're echoing back the same audio used for generation.

        Returns: (video_rgb, frame_idx, timestamp, audio_chunk_48k)
        """
        # Get next video frame (waits indefinitely for next frame)
        video_rgb, frame_idx, video_timestamp = await self.video_gen.get_next_frame()

        # Wait for next audio chunk (FIFO order) using Event-based signaling
        # The audio chunk will get the SAME timestamp as this video frame
        audio_chunk = None
        wait_start = time.time()
        timeout_seconds = 10.0  # 10 second max wait (generous for fast model)

        while audio_chunk is None:
            # Try to get audio chunk
            audio_chunk = self.audio_pass.get_next_playback_chunk()
            if audio_chunk is not None:
                break

            # Check timeout
            elapsed = time.time() - wait_start
            if elapsed > timeout_seconds:
                logger.error(f"❌ Timeout waiting for audio after {elapsed:.1f}s! "
                           f"Using silence. Queue size: {len(self.audio_pass.playback_chunks)}")
                # Create silent audio as fallback
                audio_chunk = np.zeros(self.audio_pass.chunk_size_48k, dtype=np.float32)
                break

            # Efficient wait: sleep until audio_ready Event is set
            try:
                await asyncio.wait_for(
                    self.audio_pass.audio_ready.wait(),
                    timeout=1.0  # Check every second for timeout
                )
                self.audio_pass.audio_ready.clear()  # Reset for next wait
            except asyncio.TimeoutError:
                # Periodic timeout to check overall timeout
                if elapsed > 1.0 and int(elapsed) % 2 == 0:  # Log every 2 seconds
                    available = len(self.audio_pass.playback_chunks)
                    logger.info(f"⏳ Waiting for audio for frame {frame_idx} "
                               f"(ts={video_timestamp:.3f}s), "
                               f"have {available} chunks buffered, "
                               f"waited {elapsed:.1f}s")
                continue

        self._sync_count += 1

        # Calculate actual wait time
        wait_time_ms = (time.time() - wait_start) * 1000

        # Log sync status with detailed telemetry
        if self._sync_count <= 5 or self._sync_count % 100 == 0:
            audio_queue_size = len(self.audio_pass.playback_chunks)
            video_queue_size = self.video_gen.frame_queue.qsize()

            # Status emoji based on wait time
            if wait_time_ms < 100:
                status = "✅"
            elif wait_time_ms < 500:
                status = "⚠️"
            else:
                status = "❌"

            logger.info(f"{status} Sync #{self._sync_count}: "
                       f"frame={frame_idx}, ts={video_timestamp:.3f}s, "
                       f"video_queue={video_queue_size}/200, "
                       f"audio_queue={audio_queue_size}/500, "
                       f"wait={wait_time_ms:.0f}ms")

            # Additional warning if queues are unusual
            if video_queue_size > 180:
                logger.warning(f"⚠️ Video queue near full ({video_queue_size}/200) - "
                             f"model generating faster than network can transmit")
            if audio_queue_size < 5 and self._sync_count > 25:
                logger.warning(f"⚠️ Audio queue low ({audio_queue_size}) - "
                             f"client may not be sending audio fast enough")

        # Return video with audio that gets the SAME timestamp
        return (video_rgb, frame_idx, video_timestamp, audio_chunk)


class BufferedAudioTrack(AudioStreamTrack):
    """
    WebRTC audio track that outputs synchronized audio.
    """

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.samples_per_frame = 1920  # 40ms @ 48kHz
        self.audio_queue = asyncio.Queue(maxsize=200)
        self._frame_count = 0
        self._silence_count = 0

    def add_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """Add audio chunk with timestamp."""
        try:
            # Clip to [-1, 1]
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

            # Convert to int16
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Create av.AudioFrame
            frame = av.AudioFrame(format='s16', layout='mono', samples=len(audio_int16))
            frame.sample_rate = self.sample_rate
            frame.pts = int(timestamp * self.sample_rate)
            frame.time_base = Fraction(1, self.sample_rate)

            # Copy audio data
            frame.planes[0].update(audio_int16.tobytes())

            # Add to queue
            self.audio_queue.put_nowait((frame, timestamp))

            if self._frame_count == 0:
                logger.info(f"🔊 First audio chunk queued: {len(audio_int16)} samples, "
                           f"PTS={frame.pts}, timestamp={timestamp:.3f}s")

        except asyncio.QueueFull:
            if self._frame_count % 25 == 0:
                logger.warning(f"⚠️ Audio queue full, dropping chunk")

    async def recv(self):
        """Return next audio frame for WebRTC."""
        try:
            frame, timestamp = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            self._frame_count += 1

            if self._frame_count <= 3:
                logger.info(f"🔊 Audio frame #{self._frame_count} → WebRTC: "
                           f"PTS={frame.pts} ({frame.pts/self.sample_rate:.3f}s)")

            return frame

        except asyncio.TimeoutError:
            # No audio available, send silence
            self._silence_count += 1

            silence_pts = int(self._silence_count * self.samples_per_frame)
            frame = av.AudioFrame(format='s16', layout='mono', samples=self.samples_per_frame)
            frame.sample_rate = self.sample_rate
            frame.pts = silence_pts
            frame.time_base = Fraction(1, self.sample_rate)

            # Fill with silence
            for p in frame.planes:
                p.update(bytes(p.buffer_size))

            if self._silence_count % 50 == 1:
                logger.warning(f"⚠️ Sending silence frame #{self._silence_count}")

            return frame


class BufferedVideoTrack(VideoStreamTrack):
    """
    WebRTC video track that outputs synchronized video.
    """

    def __init__(self, synchronizer: AVSynchronizer, audio_track: BufferedAudioTrack):
        super().__init__()
        self.synchronizer = synchronizer
        self.audio_track = audio_track
        self.target_fps = 25
        self._frame_count = 0

    async def recv(self):
        """Return next video frame for WebRTC."""
        # Get synchronized AV pair (waits for both video AND audio to be ready)
        video_rgb, frame_idx, timestamp, audio_chunk = await self.synchronizer.get_next()

        # Add audio to audio track with the SAME timestamp as video
        # This ensures perfect sync - audio gets video's timestamp
        self.audio_track.add_chunk(audio_chunk, timestamp)

        # Create video frame
        frame = av.VideoFrame.from_ndarray(video_rgb, format="rgb24")
        frame.pts = int(timestamp * self.target_fps)
        frame.time_base = Fraction(1, self.target_fps)

        if self._frame_count == 0:
            logger.info(f"🎬 First video frame → WebRTC: {frame.width}x{frame.height}, "
                       f"PTS={frame.pts} ({frame.pts/self.target_fps:.3f}s)")

        self._frame_count += 1

        if self._frame_count % 100 == 0:
            logger.info(f"📊 Sent {self._frame_count} video frames, "
                       f"latest PTS={frame.pts} ({frame.pts/self.target_fps:.3f}s)")

        return frame


class DittoWebRTCSession:
    """
    Manages a single WebRTC session with clean component separation.
    """

    def __init__(
        self,
        websocket: WebSocketServerProtocol,
        cfg_pkl: str,
        data_root: str,
        ditto_kwargs: Optional[dict] = None
    ):
        self.websocket = websocket
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.ditto_kwargs = ditto_kwargs or {}

        # WebRTC
        self.pc: Optional[RTCPeerConnection] = None

        # Components (clean separation of concerns)
        self.audio_passthrough: Optional[AudioPassthrough] = None
        self.video_generator: Optional[VideoGenerator] = None
        self.synchronizer: Optional[AVSynchronizer] = None
        self.audio_track: Optional[BufferedAudioTrack] = None
        self.video_track: Optional[BufferedVideoTrack] = None

        # Ditto SDK
        self.sdk: Optional[StreamSDK] = None

        # State
        self.connected = False
        self._audio_task: Optional[asyncio.Task] = None

    async def setup_ditto(self, source_path: str):
        """Initialize Ditto SDK and components."""
        logger.info(f"🎭 Initializing Ditto SDK with source: {source_path}")

        # Create SDK
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root, **self.ditto_kwargs)

        # Create components
        self.audio_passthrough = AudioPassthrough()
        self.video_generator = VideoGenerator(self.sdk, target_fps=25)  # Enforce 25 FPS
        self.synchronizer = AVSynchronizer(self.video_generator, self.audio_passthrough)
        self.audio_track = BufferedAudioTrack()
        self.video_track = BufferedVideoTrack(self.synchronizer, self.audio_track)

        # Setup SDK with frame callback
        setup_kwargs = {
            "online_mode": True,
            "N_d": -1,
        }
        setup_kwargs.update(self.ditto_kwargs)

        self.sdk.setup(
            source_path,
            output_path=None,
            frame_callback=self.video_generator.on_frame,
            **setup_kwargs
        )

        logger.info("✅ Ditto SDK initialized successfully")

    async def handle_connect(self, message: dict):
        """Handle connection request."""
        source = message.get("source")
        if not source:
            await self.send_error("Missing 'source' field")
            return

        try:
            # Initialize Ditto
            await self.setup_ditto(source)

            # Create RTCPeerConnection
            configuration = RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
            self.pc = RTCPeerConnection(configuration=configuration)

            # Setup ICE candidate handler
            @self.pc.on("icecandidate")
            async def on_icecandidate(candidate):
                if candidate:
                    await self.send_message({
                        "type": "ice-candidate",
                        "candidate": {
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        }
                    })

            # Handle incoming audio from client
            @self.pc.on("track")
            async def on_track(track):
                logger.info(f"📡 Received track: {track.kind}")

                if track.kind == "audio":
                    logger.info("🎤 Starting audio processing task")
                    self._audio_task = asyncio.create_task(self._process_audio(track))

            # Add our tracks
            self.pc.addTrack(self.video_track)
            self.pc.addTrack(self.audio_track)

            logger.info("✅ Added video and audio tracks to peer connection")

            # Setup data channel (optional)
            channel = self.pc.createDataChannel("control")

            @channel.on("message")
            def on_message(msg):
                logger.info(f"📨 Data channel message: {msg}")

            self.connected = True

            await self.send_message({
                "type": "ready",
                "message": "Server ready to receive offer"
            })

        except Exception as e:
            logger.error(f"❌ Error in handle_connect: {e}", exc_info=True)
            await self.send_error(str(e))

    async def _process_audio(self, track):
        """Process incoming audio from client."""
        logger.info("🎤 Audio processing started")
        frame_count = 0

        try:
            while True:
                frame = await track.recv()
                frame_count += 1

                # Add to audio passthrough
                chunk_ready = self.audio_passthrough.add_frame(frame)

                # If we have enough audio for model, feed it
                if chunk_ready:
                    audio_chunk_16k = self.audio_passthrough.get_16khz_chunk()
                    if audio_chunk_16k is not None:
                        # Feed to Ditto model
                        self.video_generator.feed_audio(audio_chunk_16k)

                        if frame_count % 25 == 0:
                            logger.debug(f"📤 Fed {len(audio_chunk_16k)} samples (16kHz) to model")

        except Exception as e:
            if "MediaStreamError" in str(type(e).__name__):
                logger.info("🎤 Audio track ended")
            else:
                logger.error(f"❌ Error processing audio: {e}", exc_info=True)

    async def handle_offer(self, message: dict):
        """Handle WebRTC offer from client."""
        if not self.connected or not self.pc:
            await self.send_error("Not connected. Send 'connect' message first.")
            return

        try:
            sdp = message.get("sdp")
            if not sdp:
                await self.send_error("Missing 'sdp' field")
                return

            # Set remote description
            offer = RTCSessionDescription(sdp=sdp, type="offer")
            await self.pc.setRemoteDescription(offer)

            logger.info("📞 Creating answer...")

            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            # Send answer to client
            await self.send_message({
                "type": "answer",
                "sdp": self.pc.localDescription.sdp
            })

            logger.info("✅ WebRTC handshake complete")

        except Exception as e:
            logger.error(f"❌ Error in handle_offer: {e}", exc_info=True)
            await self.send_error(str(e))

    async def handle_ice_candidate(self, message: dict):
        """Handle ICE candidate from client."""
        # Using peer reflexive discovery, so we can skip this
        logger.debug("📡 Received ICE candidate (using peer reflexive discovery)")

    async def send_message(self, message: dict):
        """Send JSON message to client."""
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")

    async def send_error(self, error_message: str):
        """Send error message to client."""
        await self.send_message({
            "type": "error",
            "message": error_message
        })

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up session...")

        # Log statistics
        if self.video_generator:
            total_gen = self.video_generator._frames_generated
            total_drop = self.video_generator._frames_dropped
            if total_gen > 0:
                logger.info(f"📊 Session statistics:")
                logger.info(f"   Frames generated by model: {total_gen}")
                if total_drop > 0:
                    logger.info(f"   Frames dropped (queue overflow): {total_drop}")
                logger.info(f"   Target FPS: {self.video_generator.target_fps}")
                logger.info(f"   Pacing: Audio-driven (waits for audio before sending video)")

        if self.synchronizer:
            logger.info(f"   Total synchronized frames: {self.synchronizer._sync_count}")

        # Cancel audio processing task
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass

        # Close SDK
        if self.sdk:
            try:
                if hasattr(self.sdk, 'audio2motion_queue'):
                    self.sdk.close()
            except Exception as e:
                logger.error(f"❌ Error closing SDK: {e}")

        # Close peer connection
        if self.pc:
            await self.pc.close()

        logger.info("✅ Session cleaned up")


class DittoSignalingServer:
    """
    WebSocket signaling server for Ditto WebRTC sessions.
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        **ditto_kwargs
    ):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.host = host
        self.port = port
        self.ditto_kwargs = ditto_kwargs

        # Active sessions
        self.sessions: dict[WebSocketServerProtocol, DittoWebRTCSession] = {}

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        logger.info(f"👤 New client connected: {websocket.remote_address}")

        # Create session
        session = DittoWebRTCSession(
            websocket,
            self.cfg_pkl,
            self.data_root,
            self.ditto_kwargs
        )
        self.sessions[websocket] = session

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "connect":
                        await session.handle_connect(data)
                    elif msg_type == "offer":
                        await session.handle_offer(data)
                    elif msg_type == "ice-candidate":
                        await session.handle_ice_candidate(data)
                    else:
                        logger.warning(f"⚠️ Unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    await session.send_error("Invalid JSON")
                except Exception as e:
                    logger.error(f"❌ Error handling message: {e}", exc_info=True)
                    await session.send_error(str(e))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"👋 Client disconnected: {websocket.remote_address}")
        finally:
            await session.cleanup()
            del self.sessions[websocket]

    async def run(self):
        """Start the signaling server."""
        logger.info(f"🚀 Starting Ditto WebRTC Signaling Server v2")
        logger.info(f"📡 Listening on {self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✅ Server running. Connect clients to ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ditto WebRTC Signaling Server v2")
    parser.add_argument("--cfg_pkl", required=True, help="Path to config pickle")
    parser.add_argument("--data_root", required=True, help="Path to model data root")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--max_size", type=int, default=1920, help="Max image dimension")
    parser.add_argument("--emo", type=int, default=4, help="Emotion (0-7)")

    args = parser.parse_args()

    server = DittoSignalingServer(
        cfg_pkl=args.cfg_pkl,
        data_root=args.data_root,
        host=args.host,
        port=args.port,
        max_size=args.max_size,
        emo=args.emo,
    )

    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
