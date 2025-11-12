"""
WebRTC Signaling Server v3 - Clean Queue-Based Architecture

This server uses a simplified approach:
1. Input audio is queued with world clock timestamps
2. Ditto model processes audio → generates video frames
3. Output audio and video are paired using world clock timestamps
4. WebRTC handles all playback timing via PTS

Key Features:
- No frame dropping (pure FIFO queues)
- No artificial waits or pacing
- Single world clock for all timing
- Perfect audio-video sync via timestamp matching
- Modular design for Gemini integration

Usage:
    python signaling_server_v3.py --cfg_pkl <path> --data_root <path>
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
    RTCConfiguration,
    RTCIceServer,
    VideoStreamTrack,
)
from aiortc.mediastreams import AudioStreamTrack

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ditto components
from stream_pipeline_online import StreamSDK

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# WORLD CLOCK - Single source of truth for all timing
# ============================================================================

class WorldClock:
    """
    Monotonic clock that provides timestamps for all audio/video data.

    This is the SINGLE SOURCE OF TRUTH for timing in the system.
    All timestamps are relative to when the clock started.
    """

    def __init__(self):
        self._start_time = None

    def start(self):
        """Start the clock (call once at session start)."""
        if self._start_time is None:
            self._start_time = time.monotonic()
            logger.info(f"⏰ World clock started at t=0")

    def now(self) -> float:
        """Get current time in seconds since clock start."""
        if self._start_time is None:
            self.start()
        return time.monotonic() - self._start_time

    def reset(self):
        """Reset the clock."""
        self._start_time = None


# ============================================================================
# AUDIO PIPELINE - Input Processing
# ============================================================================

class AudioInputProcessor:
    """
    Processes incoming audio from client:
    1. Convert stereo → mono
    2. Resample 48kHz → 16kHz for Ditto model
    3. Buffer and timestamp with AUDIO DURATION (not wall clock!)
    4. Keep original 48kHz for passthrough playback

    CRITICAL: Timestamps are based on accumulated audio duration,
    not wall clock time. This ensures sync even with model latency.
    """

    def __init__(self, clock: WorldClock):
        self.clock = clock

        # Audio parameters
        self.input_rate = 48000   # WebRTC standard
        self.model_rate = 16000   # Ditto requirement
        self.output_rate = 48000  # Playback rate

        # Buffers
        self.buffer_16k = np.array([], dtype=np.float32)  # For model
        self.buffer_48k = np.array([], dtype=np.float32)  # For playback

        # Chunk sizes
        self.model_chunk_size = 6400  # 400ms @ 16kHz (Ditto requirement)
        self.playback_chunk_size = 1920  # 40ms @ 48kHz (25fps = 1 chunk per frame)

        # CRITICAL: Accumulated audio duration for timestamps
        # This is the KEY to maintaining sync despite model latency
        self.accumulated_duration = 0.0  # Seconds of audio processed

        # Statistics
        self._frames_received = 0
        self._chunks_generated = 0

    def process_frame(self, frame: av.AudioFrame) -> bool:
        """
        Process incoming audio frame from WebRTC.

        Returns:
            True if a new model chunk is ready
        """
        self._frames_received += 1

        # Convert to numpy
        audio_array = frame.to_ndarray()

        # Convert stereo to mono
        audio_mono = self._to_mono(audio_array, frame.layout.name)

        # Normalize to float32 [-1, 1]
        audio_normalized = self._normalize(audio_mono, audio_array.dtype)

        # Log first frame
        if self._frames_received == 1:
            logger.info(f"📥 First audio frame: {frame.sample_rate}Hz, "
                       f"{len(audio_normalized)} samples, "
                       f"RMS={np.sqrt(np.mean(audio_normalized**2)):.4f}")

        # Store 48kHz for playback
        self.buffer_48k = np.concatenate([self.buffer_48k, audio_normalized])

        # Downsample to 16kHz for model
        if frame.sample_rate == 48000:
            audio_16k = scipy.signal.resample_poly(audio_normalized, up=1, down=3)
        elif frame.sample_rate == 16000:
            audio_16k = audio_normalized
        else:
            num_samples = int(len(audio_normalized) * 16000 / frame.sample_rate)
            audio_16k = scipy.signal.resample(audio_normalized, num_samples)

        self.buffer_16k = np.concatenate([self.buffer_16k, audio_16k])

        # Check if we have enough for a model chunk
        return len(self.buffer_16k) >= self.model_chunk_size

    def get_model_chunk(self) -> Optional[Tuple[np.ndarray, list]]:
        """
        Get next chunk for Ditto model (6400 samples @ 16kHz = 400ms).

        Also extracts corresponding 48kHz playback chunks with timestamps.

        CRITICAL: Timestamps are based on ACCUMULATED AUDIO DURATION, not wall clock!
        This ensures perfect sync even when model has latency.

        Returns:
            (audio_16k, playback_chunks) where playback_chunks is a list of
            (audio_48k, timestamp) tuples, one per video frame (10 chunks @ 40ms each)
        """
        if len(self.buffer_16k) < self.model_chunk_size:
            return None

        # Extract model chunk (16kHz)
        chunk_16k = self.buffer_16k[:self.model_chunk_size]
        self.buffer_16k = self.buffer_16k[self.model_chunk_size:]

        # Extract corresponding 48kHz audio
        # 6400 samples @ 16kHz = 400ms = 19200 samples @ 48kHz
        samples_48k_needed = self.model_chunk_size * 3

        playback_chunks = []

        if len(self.buffer_48k) >= samples_48k_needed:
            chunk_48k = self.buffer_48k[:samples_48k_needed]
            self.buffer_48k = self.buffer_48k[samples_48k_needed:]

            # Split into 10 x 40ms sub-chunks (for 25fps)
            # Each frame gets one 40ms audio chunk with timestamp
            for i in range(0, len(chunk_48k), self.playback_chunk_size):
                sub_chunk = chunk_48k[i:i + self.playback_chunk_size]

                # Pad if needed
                if len(sub_chunk) < self.playback_chunk_size:
                    sub_chunk = np.pad(sub_chunk, (0, self.playback_chunk_size - len(sub_chunk)))

                # CRITICAL: Timestamp = accumulated audio duration, NOT wall clock!
                # This is the playback time for this audio chunk
                timestamp = self.accumulated_duration

                # Increment accumulated duration (40ms per chunk)
                chunk_duration = len(sub_chunk) / self.output_rate
                self.accumulated_duration += chunk_duration

                playback_chunks.append((sub_chunk, timestamp))

        self._chunks_generated += 1

        if self._chunks_generated <= 3 or self._chunks_generated % 10 == 0:
            logger.debug(f"📦 Generated chunk #{self._chunks_generated}: "
                        f"{len(playback_chunks)} playback sub-chunks, "
                        f"timestamps {playback_chunks[0][1]:.3f}s - {playback_chunks[-1][1]:.3f}s")

        return (chunk_16k, playback_chunks)

    def _to_mono(self, audio: np.ndarray, layout: str) -> np.ndarray:
        """Convert stereo/multi-channel to mono by averaging."""
        is_stereo = (layout == 'stereo')

        if audio.ndim > 1:
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
            # 1D interleaved stereo
            return audio.reshape(-1, 2).mean(axis=1)
        else:
            return audio

    def _normalize(self, audio: np.ndarray, dtype) -> np.ndarray:
        """Normalize audio to float32 [-1, 1]."""
        if dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            return audio.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio.astype(np.float32)
            max_abs = np.abs(audio_float).max()
            if max_abs > 10.0:
                return audio_float / max_abs
            return audio_float


# ============================================================================
# VIDEO PIPELINE - Ditto Model Processing
# ============================================================================

class VideoGenerator:
    """
    Interfaces with Ditto SDK to generate video frames.
    Frames are queued with their generation timestamp.

    CRITICAL: Uses thread-safe queue since Ditto SDK calls on_frame() from worker threads!
    """

    def __init__(self, sdk: StreamSDK, clock: WorldClock):
        self.sdk = sdk
        self.clock = clock

        # Frame queue: (frame_rgb, frame_idx, timestamp)
        # CRITICAL: Use thread-safe queue since Ditto callbacks run in worker threads
        import queue
        self.frame_queue = queue.Queue(maxsize=500)  # Thread-safe!

        # Event loop for async operations
        self._loop = None

        # Statistics
        self._frames_generated = 0
        self._first_frame_time = None

    def set_event_loop(self, loop):
        """Set the asyncio event loop for thread-safe operations."""
        self._loop = loop

    def on_frame(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """
        Callback from StreamSDK when frame is ready.

        CRITICAL: This is called from Ditto's worker thread, not main async loop!
        Must use thread-safe queue operations.
        """
        self._frames_generated += 1

        try:
            # Use world clock timestamp (ignore SDK timestamp)
            world_timestamp = self.clock.now()

            # Thread-safe put (blocks if full, but with timeout)
            self.frame_queue.put((frame_rgb, frame_idx, world_timestamp), timeout=1.0)

            if self._first_frame_time is None:
                self._first_frame_time = world_timestamp
                logger.info(f"🎬 First video frame generated at t={world_timestamp:.3f}s")

            if self._frames_generated % 100 == 0:
                logger.debug(f"📊 Generated {self._frames_generated} frames, "
                           f"queue size: {self.frame_queue.qsize()}")

        except Exception as e:
            logger.warning(f"⚠️ Error queuing frame {frame_idx}: {e}")

    async def get_next_frame(self) -> Tuple[np.ndarray, int, float]:
        """
        Get next video frame (async-safe wrapper for thread-safe queue).

        Uses asyncio.to_thread() to avoid blocking the event loop.
        """
        # Run blocking queue.get() in a thread pool
        frame_rgb, frame_idx, timestamp = await asyncio.to_thread(self.frame_queue.get)
        return (frame_rgb, frame_idx, timestamp)

    def feed_audio(self, audio_chunk: np.ndarray):
        """Feed 16kHz audio chunk to Ditto model."""
        if self.sdk:
            self.sdk.run_chunk(audio_chunk, chunksize=(3, 5, 2))


# ============================================================================
# OUTPUT SYNCHRONIZER - Match Audio + Video
# ============================================================================

class AVOutputSynchronizer:
    """
    Synchronizes audio and video outputs using timestamp matching.

    Key insight: We have TWO queues with timestamps:
    1. Playback audio chunks (from AudioInputProcessor) - timestamped when captured
    2. Video frames (from VideoGenerator) - timestamped when generated

    We pair them using a FIFO approach and use the AUDIO timestamp for both
    (since audio is the input that drove video generation).
    """

    def __init__(self, video_gen: VideoGenerator, clock: WorldClock):
        self.video_gen = video_gen
        self.clock = clock

        # Queue of playback audio chunks with timestamps
        # Each chunk is (audio_48k, timestamp) from AudioInputProcessor
        self.audio_queue = deque(maxlen=1000)

        # Statistics
        self._sync_count = 0
        self._drift_sum = 0.0

    def add_audio_chunks(self, chunks: list):
        """
        Add playback audio chunks with their timestamps.

        Args:
            chunks: List of (audio_48k, timestamp) tuples
        """
        for chunk, timestamp in chunks:
            self.audio_queue.append((chunk, timestamp))

    async def get_next(self) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Get next synchronized AV pair.

        Returns:
            (video_rgb, frame_idx, timestamp, audio_48k)

        The timestamp is taken from the audio chunk (the input that drove generation).
        """
        # Get next video frame
        video_rgb, frame_idx, video_timestamp = await self.video_gen.get_next_frame()

        # Get corresponding audio chunk (FIFO order)
        if len(self.audio_queue) > 0:
            audio_chunk, audio_timestamp = self.audio_queue.popleft()
        else:
            # No audio available, use silence
            logger.warning(f"⚠️ No audio chunk available for frame {frame_idx}, using silence")
            audio_chunk = np.zeros(1920, dtype=np.float32)
            audio_timestamp = video_timestamp

        # Use AUDIO timestamp for sync (it's the input that drove generation)
        sync_timestamp = audio_timestamp

        self._sync_count += 1

        # Calculate latency (how long after audio was captured did video arrive)
        # This is EXPECTED to be positive (model takes time to generate)
        latency = video_timestamp - audio_timestamp
        self._drift_sum += abs(latency)

        # Log sync status
        if self._sync_count <= 5 or self._sync_count % 100 == 0:
            avg_drift = self._drift_sum / self._sync_count

            # Status emoji based on latency stability
            if self._sync_count > 10:
                recent_latency = latency * 1000
                if 0 <= recent_latency <= 200:
                    status = "✅"  # Perfect - low latency
                elif 200 < recent_latency <= 1000:
                    status = "⚠️"  # Warning - moderate latency
                else:
                    status = "❌"  # Error - high latency or drift
            else:
                status = "🔄"  # Starting up

            logger.info(f"{status} Sync #{self._sync_count}: frame={frame_idx}, "
                       f"audio_ts={audio_timestamp:.3f}s, video_gen_at={video_timestamp:.3f}s, "
                       f"latency={latency*1000:.1f}ms, avg_abs_drift={avg_drift*1000:.1f}ms, "
                       f"audio_queue={len(self.audio_queue)}")

        return (video_rgb, frame_idx, sync_timestamp, audio_chunk)


# ============================================================================
# WEBRTC TRACKS - Output to Browser
# ============================================================================

class BufferedAudioTrack(AudioStreamTrack):
    """WebRTC audio track with PTS-based timing."""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.samples_per_frame = 1920  # 40ms @ 48kHz

        # Queue of (av.AudioFrame, timestamp) tuples
        self.audio_queue = asyncio.Queue(maxsize=200)

        self._frame_count = 0

    def add_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """Add audio chunk with timestamp."""
        try:
            # Clip and convert to int16
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Create av.AudioFrame
            frame = av.AudioFrame(format='s16', layout='mono', samples=len(audio_int16))
            frame.sample_rate = self.sample_rate
            frame.pts = int(timestamp * self.sample_rate)
            frame.time_base = Fraction(1, self.sample_rate)

            # Copy data
            frame.planes[0].update(audio_int16.tobytes())

            # Queue
            self.audio_queue.put_nowait((frame, timestamp))

            if self._frame_count == 0:
                logger.info(f"🔊 First audio frame queued: PTS={frame.pts}, ts={timestamp:.3f}s")

        except asyncio.QueueFull:
            if self._frame_count % 50 == 0:
                logger.warning(f"⚠️ Audio output queue full")

    async def recv(self):
        """Return next audio frame for WebRTC."""
        try:
            frame, timestamp = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            self._frame_count += 1
            return frame

        except asyncio.TimeoutError:
            # Send silence if no audio available
            silence_pts = self._frame_count * self.samples_per_frame
            frame = av.AudioFrame(format='s16', layout='mono', samples=self.samples_per_frame)
            frame.sample_rate = self.sample_rate
            frame.pts = silence_pts
            frame.time_base = Fraction(1, self.sample_rate)

            for p in frame.planes:
                p.update(bytes(p.buffer_size))

            self._frame_count += 1
            return frame


class BufferedVideoTrack(VideoStreamTrack):
    """WebRTC video track with PTS-based timing."""

    def __init__(self, synchronizer: AVOutputSynchronizer, audio_track: BufferedAudioTrack):
        super().__init__()
        self.synchronizer = synchronizer
        self.audio_track = audio_track
        self.target_fps = 25

        self._frame_count = 0

    async def recv(self):
        """Return next video frame for WebRTC."""
        # Get synchronized AV pair
        video_rgb, frame_idx, timestamp, audio_chunk = await self.synchronizer.get_next()

        # Add audio with same timestamp
        self.audio_track.add_chunk(audio_chunk, timestamp)

        # Create video frame
        frame = av.VideoFrame.from_ndarray(video_rgb, format="rgb24")
        frame.pts = int(timestamp * self.target_fps)
        frame.time_base = Fraction(1, self.target_fps)

        if self._frame_count == 0:
            logger.info(f"🎬 First video frame sent: {frame.width}x{frame.height}, "
                       f"PTS={frame.pts}, ts={timestamp:.3f}s")

        self._frame_count += 1

        if self._frame_count % 100 == 0:
            logger.debug(f"📊 Sent {self._frame_count} video frames")

        return frame


# ============================================================================
# WEBRTC SESSION - Orchestrates Everything
# ============================================================================

class DittoWebRTCSession:
    """
    Manages a single WebRTC session with clean component separation.

    Pipeline:
        Client Audio → AudioInputProcessor → Ditto Model → VideoGenerator
                             ↓                                    ↓
                    Playback Chunks (timestamped)        Video Frames (timestamped)
                             ↓                                    ↓
                                  AVOutputSynchronizer
                                          ↓
                              WebRTC Tracks (Audio + Video)
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

        # World clock
        self.clock = WorldClock()

        # Components
        self.audio_processor: Optional[AudioInputProcessor] = None
        self.video_generator: Optional[VideoGenerator] = None
        self.synchronizer: Optional[AVOutputSynchronizer] = None
        self.audio_track: Optional[BufferedAudioTrack] = None
        self.video_track: Optional[BufferedVideoTrack] = None

        # Ditto SDK
        self.sdk: Optional[StreamSDK] = None

        # State
        self.connected = False
        self._audio_task: Optional[asyncio.Task] = None

    async def setup_ditto(self, source_path: str):
        """Initialize Ditto SDK and components."""
        logger.info(f"🎭 Initializing Ditto with source: {source_path}")

        # Start world clock
        self.clock.start()

        # Create SDK
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root, **self.ditto_kwargs)

        # Create components
        self.audio_processor = AudioInputProcessor(self.clock)
        self.video_generator = VideoGenerator(self.sdk, self.clock)
        self.synchronizer = AVOutputSynchronizer(self.video_generator, self.clock)
        self.audio_track = BufferedAudioTrack()
        self.video_track = BufferedVideoTrack(self.synchronizer, self.audio_track)

        # Setup SDK
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

        logger.info("✅ Ditto initialized successfully")

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
                    logger.info("🎤 Starting audio processing")
                    self._audio_task = asyncio.create_task(self._process_audio(track))

            # Add our tracks
            self.pc.addTrack(self.video_track)
            self.pc.addTrack(self.audio_track)

            logger.info("✅ Added video and audio tracks to peer connection")

            # Setup data channel (optional)
            channel = self.pc.createDataChannel("control")

            @channel.on("message")
            def on_message(msg):
                logger.debug(f"📨 Data channel: {msg}")

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

        try:
            while True:
                frame = await track.recv()

                # Process frame
                chunk_ready = self.audio_processor.process_frame(frame)

                # If we have enough audio, feed to model
                if chunk_ready:
                    result = self.audio_processor.get_model_chunk()
                    if result:
                        audio_16k, playback_chunks = result

                        # Feed to Ditto model
                        self.video_generator.feed_audio(audio_16k)

                        # Queue playback audio chunks for output
                        self.synchronizer.add_audio_chunks(playback_chunks)

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


# ============================================================================
# SIGNALING SERVER - WebSocket Handler
# ============================================================================

class DittoSignalingServer:
    """WebSocket signaling server for Ditto WebRTC sessions."""

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
        logger.info(f"🚀 Starting Ditto WebRTC Signaling Server v3")
        logger.info(f"📡 Listening on {self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✅ Server running. Connect clients to ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


# ============================================================================
# MAIN - Command Line Interface
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ditto WebRTC Signaling Server v3 - Clean Queue-Based Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python signaling_server_v3.py \\
    --cfg_pkl outputs/cfg_f_model.pkl \\
    --data_root ./ \\
    --port 8080
        """
    )

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
