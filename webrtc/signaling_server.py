"""
WebSocket Signaling Server for WebRTC

This server handles WebRTC signaling (SDP offer/answer exchange and ICE candidates)
between web clients and the Ditto avatar server using aiortc.

Protocol:
    Client -> Server:
        {"type": "connect", "source": "path/to/avatar.jpg"}
        {"type": "offer", "sdp": "..."}
        {"type": "ice-candidate", "candidate": {...}}

    Server -> Client:
        {"type": "answer", "sdp": "..."}
        {"type": "ice-candidate", "candidate": {...}}
        {"type": "error", "message": "..."}
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List
from fractions import Fraction
import numpy as np
import scipy.signal
import websockets
from websockets.legacy.server import WebSocketServerProtocol

# WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import AudioStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import av

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ditto components
from stream_pipeline_online import StreamSDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BufferedAudioTrack(AudioStreamTrack):
    """
    Audio track that plays back buffered audio synchronized with video generation.
    Stores original av.AudioFrame objects with timestamps matching video.
    This is the AUDIO CLOCK - video synchronization is driven by this track's playback time.
    """

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000  # WebRTC/Opus prefers 48kHz for proper browser playback
        self.ditto_rate = 16000   # Ditto's native rate
        self.audio_queue = asyncio.Queue(maxsize=200)  # Buffer for (av.AudioFrame, timestamp) tuples
        self._samples_per_frame = 1920  # WebRTC frame size at 48kHz (40ms at 48kHz = 1920 samples)

        # Audio clock: tracks current playback time in seconds
        self._playback_start_time = None  # When audio playback started (wall clock)
        self._playback_time_offset = 0.0  # Current audio time in seconds

    def add_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """Convert numpy audio chunk to av.AudioFrame and add to buffer with timestamp."""
        try:
            # Log first chunk
            if not hasattr(self, '_chunk_count'):
                self._chunk_count = 0
                rms = np.sqrt(np.mean(audio_chunk**2))
                logger.warning(f"🎵 FIRST AUDIO CHUNK:")
                logger.warning(f"   Samples: {len(audio_chunk)} (expected 1920 for 40ms@48kHz)")
                logger.warning(f"   Sample rate: {self.sample_rate}Hz")
                logger.warning(f"   Timestamp: {timestamp:.3f}s")
                logger.warning(f"   RMS: {rms:.4f}")

                # CRITICAL: Verify this is 48kHz audio, not 16kHz!
                # At 48kHz, 40ms = 1920 samples
                # At 16kHz, 40ms = 640 samples
                if len(audio_chunk) == 640:
                    logger.error("⚠️⚠️⚠️ BUG: Received 640 samples (16kHz) but claiming 48kHz! LOW PITCH!")
                    logger.error("⚠️ This will play 3x slower = low pitch audio!")
                elif len(audio_chunk) == 1920:
                    logger.warning("✓ Correct: 1920 samples = 40ms @ 48kHz")
                else:
                    logger.warning(f"⚠️ Unexpected sample count: {len(audio_chunk)}")

            self._chunk_count += 1

            # Convert numpy array (48kHz, float32) to av.AudioFrame
            # Ensure audio is in correct range [-1, 1]
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

            # Convert float32 to int16 for av.AudioFrame
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Create av.AudioFrame with timestamp matching video
            frame = av.AudioFrame(format='s16', layout='mono', samples=len(audio_int16))
            frame.sample_rate = self.sample_rate  # 48000 Hz

            # Set PTS based on audio timestamp (same as video!)
            # Convert seconds to samples at 48kHz
            frame.pts = int(timestamp * self.sample_rate)
            frame.time_base = Fraction(1, self.sample_rate)

            # Copy audio data (mono = single plane)
            frame.planes[0].update(audio_int16.tobytes())

            # Store frame with timestamp
            self.audio_queue.put_nowait((frame, timestamp))

            # Log every 25 chunks (once per second at 25fps)
            if self._chunk_count % 25 == 0:
                logger.info(f"✓ Added {self._chunk_count} audio chunks, PTS={frame.pts}, timestamp={timestamp:.3f}s, queue size: {self.audio_queue.qsize()}")
        except asyncio.QueueFull:
            # Only log occasionally to avoid spam
            if not hasattr(self, '_drop_count'):
                self._drop_count = 0
            self._drop_count += 1
            if self._drop_count % 10 == 1:  # Log every 10th drop
                logger.warning(f"Audio buffer full, dropped {self._drop_count} chunks total")

    def get_audio_clock_time(self) -> float:
        """
        Get current audio playback time in seconds.
        This is the MASTER CLOCK for audio-video synchronization.
        """
        if self._playback_start_time is None:
            return 0.0

        # Audio time = accumulated time from frames played
        return self._playback_time_offset

    async def recv(self):
        """Return audio frames with synchronized PTS timestamps."""
        try:
            # Get audio frame and timestamp from queue
            frame, timestamp = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)

            # Initialize playback clock on first frame
            if self._playback_start_time is None:
                self._playback_start_time = time.time()
                logger.warning(f"🎵 AUDIO PLAYBACK STARTED at wall time {self._playback_start_time:.3f}")
                logger.warning(f"   First frame PTS: {frame.pts}, timestamp: {timestamp:.3f}s")

            # Log first frame with detailed info
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
                logger.warning(f"🔊 FIRST AUDIO FRAME SENT TO CLIENT:")
                logger.warning(f"   Format: {frame.format.name}")
                logger.warning(f"   Layout: {frame.layout.name}")
                logger.warning(f"   Sample rate: {frame.sample_rate}Hz")
                logger.warning(f"   Samples: {frame.samples}")
                logger.warning(f"   PTS: {frame.pts} (= {frame.pts / frame.sample_rate:.3f}s)")
                logger.warning(f"   Time base: {frame.time_base}")
                logger.warning(f"   Duration: {frame.samples / frame.sample_rate * 1000:.1f}ms")
                logger.warning(f"   Timestamp: {timestamp:.3f}s")

            self._frame_count += 1

            # Log first 5 frames in detail
            if self._frame_count <= 5:
                logger.warning(f"🔊 Audio frame #{self._frame_count} → Client:")
                logger.warning(f"   {frame.layout.name}, {frame.samples} samples @ {frame.sample_rate}Hz")
                logger.warning(f"   PTS={frame.pts} (playback at {frame.pts / frame.sample_rate:.3f}s)")
                logger.warning(f"   Duration: {frame.samples / frame.sample_rate * 1000:.1f}ms")

            # Log every 100 frames
            if self._frame_count % 100 == 0:
                logger.info(f"✓ Sent {self._frame_count} audio frames, PTS={frame.pts} ({frame.pts / frame.sample_rate:.3f}s), timestamp={timestamp:.3f}s, queue: {self.audio_queue.qsize()}")

            # Update audio playback clock based on timestamp
            self._playback_time_offset = timestamp

            # Return the frame with correct PTS
            return frame

        except asyncio.TimeoutError:
            # No audio available, send silence
            # PTS should continue from last timestamp
            silence_pts = int(self._playback_time_offset * self.sample_rate)

            frame = av.AudioFrame(format='s16', layout='mono', samples=self._samples_per_frame)
            frame.sample_rate = self.sample_rate  # 48000 Hz
            frame.pts = silence_pts
            frame.time_base = Fraction(1, self.sample_rate)

            # Fill with silence
            for p in frame.planes:
                p.update(bytes(p.buffer_size))

            # Update playback time
            frame_duration = self._samples_per_frame / self.sample_rate
            self._playback_time_offset += frame_duration

            # Log occasionally
            if not hasattr(self, '_silence_count'):
                self._silence_count = 0
                logger.warning(f"⚠ FIRST SILENCE FRAME SENT TO CLIENT:")
                logger.warning(f"   Format: {frame.format.name}")
                logger.warning(f"   Layout: {frame.layout.name}")
                logger.warning(f"   Sample rate: {frame.sample_rate}Hz")
                logger.warning(f"   Samples: {frame.samples}")
                logger.warning(f"   PTS: {frame.pts} (= {frame.pts / frame.sample_rate:.3f}s)")
                logger.warning(f"   Duration: {frame.samples / frame.sample_rate * 1000:.1f}ms")
            self._silence_count += 1
            if self._silence_count % 50 == 1:
                logger.warning(f"⚠ Sending silence frame #{self._silence_count} (audio queue empty)")

            return frame


class DittoVideoTrack(VideoStreamTrack):
    """
    Custom video track that outputs frames from Ditto pipeline.
    """

    def __init__(self, sdk: StreamSDK, buffered_audio_track=None):
        super().__init__()
        self.sdk = sdk
        self.frame_queue = asyncio.Queue(maxsize=50)  # Increased to handle recording bursts
        self._frame_count = 0
        self._first_frame_time = None
        self._last_log_time = time.time()
        self.buffered_audio_track = buffered_audio_track
        self._session_start_time = None  # Track when first frame arrives
        self._target_fps = 25

        # No artificial pacing needed - WebRTC handles timing via PTS
        # Model generates ~50 FPS, WebRTC will consume at 25 FPS based on timestamps

    def on_frame(self, frame_rgb, frame_idx, timestamp):
        """Callback from StreamSDK when frame is ready."""
        try:
            # Add current time for latency tracking
            receive_time = time.time()

            # Non-blocking put
            self.frame_queue.put_nowait((frame_rgb, timestamp, receive_time, frame_idx))

            # Log frame generation stats occasionally
            if self._first_frame_time is None:
                self._first_frame_time = receive_time
                logger.warning(f"🎬 FIRST VIDEO FRAME GENERATED at frame_idx={frame_idx}, timestamp={timestamp:.3f}s")
                logger.warning(f"   Time: {receive_time:.3f}")

                # Log delay from first audio chunk if available
                if hasattr(self.sdk, 'session') and hasattr(self.sdk.session, 'first_audio_chunk_time'):
                    delay = receive_time - self.sdk.session.first_audio_chunk_time
                    logger.warning(f"   Delay from first audio: {delay:.3f}s")

        except asyncio.QueueFull:
            # Drop frame if queue is full
            logger.warning(f"Frame queue full, dropping frame {frame_idx}")

    async def recv(self):
        """
        Receive next video frame (called by WebRTC).

        SIMPLIFIED AUDIO-DRIVEN SYNCHRONIZATION for fast model:
        - Model generates ~50 FPS (2x faster than needed)
        - We pace output to 25 FPS to match audio playback
        - Audio chunks consumed 1-per-frame to maintain sync
        """
        from av import VideoFrame
        from fractions import Fraction
        import asyncio

        # Wait for frame from Ditto pipeline
        frame_rgb, timestamp, receive_time, frame_idx = await self.frame_queue.get()

        # Initialize session start time on first frame
        if self._session_start_time is None:
            self._session_start_time = time.time()
            logger.info(f"🎬 Session started at frame {frame_idx}, timestamp {timestamp:.3f}s")

            # Log timing delay from first audio to first video
            if hasattr(self.sdk, 'session') and hasattr(self.sdk.session, 'first_audio_chunk_time'):
                delay = self._session_start_time - self.sdk.session.first_audio_chunk_time
                logger.warning(f"⏱️ TIMING: First video frame at t={self._session_start_time:.3f}, model latency: {delay:.3f}s")

        # === AUDIO-VIDEO SYNC: Let WebRTC handle pacing via PTS ===
        audio_timestamp = None
        if self.buffered_audio_track and hasattr(self.sdk, 'session'):
            session = self.sdk.session

            # Get next audio chunk with its timestamp
            if len(session.audio_chunks_queue) > 0:
                audio_chunk_data, audio_timestamp = session.audio_chunks_queue.pop(0)

                # Add audio to playback buffer WITH timestamp
                # Both audio and video will have matching PTS now!
                self.buffered_audio_track.add_audio_chunk(audio_chunk_data, audio_timestamp)
                session.chunks_retrieved += 1

                # Log audio consumption
                if self._frame_count % 50 == 0:
                    audio_clock = self.buffered_audio_track.get_audio_clock_time()
                    logger.info(f"📤 Frame {frame_idx}: audio_ts={audio_timestamp:.3f}s, audio_clock={audio_clock:.3f}s, queue_remaining={len(session.audio_chunks_queue)}")

            else:
                # No audio chunk available
                if self._frame_count > 10 and self._frame_count % 25 == 0:
                    logger.warning(f"⚠️ No audio chunk for frame {frame_idx} (queue empty)")

        # NO ARTIFICIAL PACING!
        # WebRTC will handle frame pacing based on PTS timestamps.
        # The PTS is set to audio_timestamp which has the correct timing.
        # Adding sleep here causes audio to play ahead of video!

        # Convert numpy array to av.VideoFrame
        frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        # Set PTS based on audio timestamp for proper synchronization
        # WebRTC will pace playback based on these timestamps
        if audio_timestamp is not None:
            frame.pts = int(audio_timestamp * self._target_fps)
        else:
            frame.pts = int(timestamp * self._target_fps)

        frame.time_base = Fraction(1, self._target_fps)

        # Log first video frame with detailed info
        if self._frame_count == 0:
            logger.warning(f"🎬 FIRST VIDEO FRAME SENT TO CLIENT:")
            logger.warning(f"   Resolution: {frame.width}x{frame.height}")
            logger.warning(f"   Format: {frame.format.name}")
            logger.warning(f"   PTS: {frame.pts} (= {frame.pts / self._target_fps:.3f}s @ {self._target_fps}fps)")
            logger.warning(f"   Time base: {frame.time_base}")
            logger.warning(f"   Audio timestamp: {audio_timestamp:.3f}s" if audio_timestamp else "   Audio timestamp: None")
            logger.warning(f"   Video timestamp: {timestamp:.3f}s")
            logger.warning(f"   Frame index: {frame_idx}")

        self._frame_count += 1

        # Log first 5 frames in detail
        if self._frame_count <= 5:
            logger.warning(f"🎬 Video frame #{self._frame_count} → Client:")
            logger.warning(f"   PTS={frame.pts} (playback at {frame.pts / self._target_fps:.3f}s)")
            logger.warning(f"   Audio PTS would be: {int(audio_timestamp * 48000) if audio_timestamp else 'N/A'} (playback at {audio_timestamp:.3f}s)" if audio_timestamp else "   No audio chunk")
            logger.warning(f"   Sync: Video@{frame.pts / self._target_fps:.3f}s == Audio@{audio_timestamp:.3f}s" if audio_timestamp else "   Sync: No audio chunk")

        # Periodic status logging
        if self._frame_count % 100 == 0:
            logger.info(f"📊 STATUS: Sent {self._frame_count} video frames to WebRTC, latest PTS={frame.pts} ({frame.pts / self._target_fps:.3f}s)")

        return frame


class DittoWebRTCSession:
    """
    Manages a single WebRTC session with Ditto avatar.
    """

    def __init__(
        self,
        websocket: WebSocketServerProtocol,
        cfg_pkl: str,
        data_root: str,
        ditto_kwargs: Optional[Dict] = None
    ):
        self.websocket = websocket
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.ditto_kwargs = ditto_kwargs or {}

        # WebRTC
        self.pc: Optional[RTCPeerConnection] = None
        self.video_track: Optional[DittoVideoTrack] = None
        self.relay = MediaRelay()
        self.audio_transceiver = None
        self.buffered_audio_track: Optional[BufferedAudioTrack] = None

        # Ditto SDK
        self.sdk: Optional[StreamSDK] = None
        self.source_path: Optional[str] = None

        # State
        self.connected = False

        # Timing tracking
        self.audio_chunk_count = 0
        self.last_audio_time = None
        self.audio_capture_start_time = None  # When first audio was captured (for timestamps)
        self.accumulated_audio_duration = 0.0  # Total audio duration captured in seconds

        # Audio-video synchronization - TIMESTAMP-BASED APPROACH
        # Queue of (audio_chunk, timestamp) tuples where timestamp is the audio's capture time in seconds
        self.audio_chunks_queue = []  # List of (np.ndarray, float) tuples
        self.chunks_stored = 0  # Total sub-chunks stored
        self.chunks_retrieved = 0  # Total sub-chunks retrieved

    async def setup_ditto(self, source_path: str):
        """Initialize Ditto SDK with avatar source."""
        self.source_path = source_path

        logger.info(f"Initializing Ditto SDK with source: {source_path}")

        # Create SDK
        self.sdk = StreamSDK(
            self.cfg_pkl,
            self.data_root,
            **self.ditto_kwargs
        )

        # Store session reference in SDK for accessing frame_audio_map
        self.sdk.session = self

        # Create video track with buffered audio track reference
        # (Will be set later in handle_connect, so for now pass None)
        self.video_track = DittoVideoTrack(self.sdk, self.buffered_audio_track)

        # Setup SDK with frame callback
        setup_kwargs = {
            "online_mode": True,
            "N_d": -1,
        }
        setup_kwargs.update(self.ditto_kwargs)

        self.sdk.setup(
            source_path,
            output_path=None,
            frame_callback=self.video_track.on_frame,
            **setup_kwargs
        )

        logger.info("Ditto SDK initialized successfully")

    async def handle_connect(self, message: dict):
        """Handle connection request with avatar source."""
        source = message.get("source")
        if not source:
            await self.send_error("Missing 'source' field")
            return

        try:
            # Initialize Ditto
            await self.setup_ditto(source)

            # Create RTCPeerConnection
            configuration = RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                    # Add TURN servers if needed for NAT traversal
                ]
            )
            self.pc = RTCPeerConnection(configuration=configuration)

            # Create buffered audio track BEFORE setting up handlers
            # (so it exists when audio frames start arriving)
            self.buffered_audio_track = BufferedAudioTrack()
            logger.info("Created buffered audio track for synchronized playback")

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

            # Handle incoming audio track from client
            @self.pc.on("track")
            async def on_track(track):
                logger.info(f"Received track: {track.kind}")

                if track.kind == "audio":
                    logger.info("Starting audio processing (audio will be buffered and synced with video)")

                    # Audio buffer for accumulation
                    # chunksize=(3,5,2) means 10 frames total, each frame is 640 samples (40ms @ 16kHz)
                    audio_buffer = np.array([], dtype=np.float32)
                    min_chunk_size = 6400  # 10 frames * 640 samples = 400ms of audio
                    first_frame_logged = False

                    # Process audio frames
                    while True:
                        try:
                            frame = await track.recv()

                            # Don't add immediately - will be synced with video later
                            # Convert audio frame to raw PCM for Ditto processing
                            # frame is av.AudioFrame

                            # Log first frame info
                            if not first_frame_logged:
                                logger.info(f"First audio frame: format={frame.format.name}, layout={frame.layout.name}, sample_rate={frame.sample_rate}, samples={frame.samples}")
                                first_frame_logged = True

                            # Convert to numpy array
                            audio_array = frame.to_ndarray()

                            if not hasattr(self, '_first_array_logged'):
                                logger.warning(f"🔍 First audio array: dtype={audio_array.dtype}, shape={audio_array.shape}, range=[{audio_array.min()}, {audio_array.max()}]")
                                logger.warning(f"   Frame: layout={frame.layout.name}, samples={frame.samples}, rate={frame.sample_rate}Hz")
                                self._first_array_logged = True

                            # Convert stereo to mono by AVERAGING channels (not flattening!)
                            # CRITICAL: flatten() would interleave L/R causing 2x duration error

                            # Check if frame is stereo based on layout
                            is_stereo = (frame.layout.name == 'stereo')

                            if audio_array.ndim > 1:
                                # Stereo audio: shape is (channels, samples) or (samples, channels)
                                if audio_array.shape[0] == 2:  # (2, N) - channels first (planar)
                                    logger.warning(f"   Stereo planar (2, {audio_array.shape[1]}): averaging channels")
                                    audio_array = audio_array.mean(axis=0)  # Average L and R channels
                                elif audio_array.shape[1] == 2:  # (N, 2) - samples first (planar)
                                    logger.warning(f"   Stereo planar ({audio_array.shape[0]}, 2): averaging channels")
                                    audio_array = audio_array.mean(axis=1)  # Average L and R channels
                                elif is_stereo and audio_array.shape[0] == 1:
                                    # Interleaved stereo: (1, N) where N = 2*samples
                                    # De-interleave [L1,R1,L2,R2,...] and average
                                    logger.warning(f"   Stereo INTERLEAVED {audio_array.shape}: de-interleaving and averaging")
                                    audio_flat = audio_array[0]  # Get the 1D array
                                    # Reshape to (samples, 2) where each row is [L, R]
                                    audio_deinterleaved = audio_flat.reshape(-1, 2)
                                    audio_array = audio_deinterleaved.mean(axis=1)  # Average L and R
                                    logger.warning(f"   De-interleaved {len(audio_flat)} → {len(audio_array)} mono samples")
                                else:
                                    # Multi-channel or unknown layout - take first channel
                                    logger.warning(f"   Multi-channel audio {audio_array.shape}, taking first channel")
                                    audio_array = audio_array[0] if audio_array.shape[0] < audio_array.shape[1] else audio_array[:, 0]
                            elif is_stereo and audio_array.ndim == 1:
                                # 1D interleaved stereo: [L1,R1,L2,R2,...]
                                logger.warning(f"   Stereo INTERLEAVED 1D ({len(audio_array)}): de-interleaving and averaging")
                                audio_deinterleaved = audio_array.reshape(-1, 2)
                                audio_array = audio_deinterleaved.mean(axis=1)
                                logger.warning(f"   De-interleaved {len(audio_array)*2} → {len(audio_array)} mono samples")

                            if not hasattr(self, '_stereo_converted_logged'):
                                logger.warning(f"   ✓ After stereo→mono conversion: {audio_array.shape}")
                                self._stereo_converted_logged = True

                            # Convert to float32 and normalize
                            # WebRTC typically sends int16 audio, need to normalize to [-1, 1]
                            if audio_array.dtype == np.int16:
                                audio_float = audio_array.astype(np.float32) / 32768.0
                            elif audio_array.dtype == np.int32:
                                audio_float = audio_array.astype(np.float32) / 2147483648.0
                            else:
                                # Already float, might already be normalized
                                audio_float = audio_array.astype(np.float32)
                                # Check if normalized
                                max_abs = np.abs(audio_float).max()
                                if max_abs > 10.0:  # Definitely not normalized
                                    logger.warning(f"Audio appears unnormalized (max={max_abs}), normalizing...")
                                    audio_float = audio_float / max_abs

                            # Store original 48kHz audio for passthrough playback
                            audio_48k = audio_float.copy()
                            original_sample_rate = frame.sample_rate

                            # Debug: log audio stats on first frame
                            if not hasattr(self, '_audio_debug_logged'):
                                self._audio_debug_logged = True
                                logger.info(f"🎤 FIRST AUDIO FROM CLIENT:")
                                logger.info(f"   Sample rate: {frame.sample_rate}Hz")
                                logger.info(f"   Samples: {len(audio_float)}")
                                logger.info(f"   Format: {frame.format.name}, Layout: {frame.layout.name}")
                                logger.info(f"   audio_48k has {len(audio_48k)} samples")

                            # Downsample to 16kHz for Ditto processing only
                            if frame.sample_rate == 48000:
                                # 48kHz → 16kHz using polyphase filter (up=1, down=3)
                                audio_16k = scipy.signal.resample_poly(audio_float, up=1, down=3)
                                logger.info(f"✓ Downsampled 48000Hz → 16000Hz, {len(audio_16k)} samples")
                            elif frame.sample_rate == 16000:
                                audio_16k = audio_float
                                logger.info(f"✓ Audio already at 16000Hz, {len(audio_16k)} samples")
                            else:
                                # Generic resampling for other rates
                                num_samples = int(len(audio_float) * 16000 / frame.sample_rate)
                                audio_16k = scipy.signal.resample(audio_float, num_samples)
                                logger.info(f"✓ Resampled {frame.sample_rate}Hz → 16000Hz, {len(audio_16k)} samples")

                            # Accumulate 16kHz audio for Ditto processing
                            audio_buffer = np.concatenate([audio_buffer, audio_16k])

                            # Accumulate 48kHz audio for playback
                            if not hasattr(self, 'audio_buffer_48k'):
                                self.audio_buffer_48k = np.array([], dtype=np.float32)
                            self.audio_buffer_48k = np.concatenate([self.audio_buffer_48k, audio_48k])

                            # Debug: log audio stats
                            if len(audio_buffer) >= min_chunk_size:
                                logger.info(f"Audio buffer: {len(audio_buffer)} samples, RMS: {np.sqrt(np.mean(audio_float**2)):.4f}")

                            # Send to SDK when we have enough samples
                            while len(audio_buffer) >= min_chunk_size:
                                # Take min_chunk_size samples
                                chunk = audio_buffer[:min_chunk_size]
                                audio_buffer = audio_buffer[min_chunk_size:]

                                # Normalize if needed (WebRTC audio is usually already normalized to [-1, 1])
                                # But check if values are too large
                                max_val = np.abs(chunk).max()
                                if max_val > 1.0:
                                    logger.warning(f"Audio not normalized! Max value: {max_val}, normalizing...")
                                    chunk = chunk / max_val

                                # Track timing
                                chunk_time = time.time()
                                self.audio_chunk_count += 1

                                # Track timing for first chunk
                                if self.audio_chunk_count == 1:
                                    self.first_audio_chunk_time = chunk_time
                                    logger.warning(f"⏱️ TIMING: First audio chunk sent to Ditto at t={chunk_time:.3f}")

                                logger.info(f"Sending audio chunk #{self.audio_chunk_count}: {len(chunk)} samples (16kHz), RMS: {np.sqrt(np.mean(chunk**2)):.4f}, range: [{chunk.min():.4f}, {chunk.max():.4f}]")

                                # Feed to SDK (16kHz audio for Ditto processing)
                                if self.sdk:
                                    self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))
                                    self.last_audio_time = chunk_time

                                    # Store 48kHz audio chunks in FIFO queue for passthrough playback
                                    # Extract corresponding 48kHz audio (3x the samples since 48k/16k = 3)
                                    # At 25fps: 40ms per frame = 640 samples @ 16kHz = 1920 samples @ 48kHz
                                    chunk_48k_size = 1920  # 40ms @ 48kHz

                                    # Get 48kHz audio for this chunk
                                    # We processed 6400 samples @ 16kHz, which corresponds to 19200 samples @ 48kHz
                                    samples_48k_needed = len(chunk) * 3  # 16kHz → 48kHz is 3x

                                    if len(self.audio_buffer_48k) >= samples_48k_needed:
                                        chunk_48k = self.audio_buffer_48k[:samples_48k_needed]
                                        self.audio_buffer_48k = self.audio_buffer_48k[samples_48k_needed:]

                                        # Split into 25fps chunks (1920 samples each @ 48kHz)
                                        start_queue_idx = self.chunks_stored

                                        # CRITICAL: Verify we're splitting the RIGHT audio (48kHz, not 16kHz)
                                        if self.chunks_stored == 0:
                                            logger.warning(f"🔍 FIRST STORAGE - VERIFYING AUDIO SOURCE:")
                                            logger.warning(f"   chunk_48k length: {len(chunk_48k)} samples")
                                            logger.warning(f"   Expected: {samples_48k_needed} (6400 * 3 = 19200)")
                                            logger.warning(f"   Will split into {len(chunk_48k) // chunk_48k_size} chunks of {chunk_48k_size}")
                                            logger.warning(f"   Source: audio_buffer_48k (NOT the 16kHz buffer)")

                                        # Initialize audio capture clock on first chunk
                                        if self.audio_capture_start_time is None:
                                            self.audio_capture_start_time = time.time()
                                            logger.warning(f"🎤 AUDIO CAPTURE STARTED at wall time {self.audio_capture_start_time:.3f}")

                                        for i in range(0, len(chunk_48k), chunk_48k_size):
                                            sub_chunk_48k = chunk_48k[i:i+chunk_48k_size]

                                            # Pad if needed (last chunk might be shorter)
                                            if len(sub_chunk_48k) < chunk_48k_size:
                                                sub_chunk_48k = np.pad(sub_chunk_48k, (0, chunk_48k_size - len(sub_chunk_48k)))

                                            rms = np.sqrt(np.mean(sub_chunk_48k**2))

                                            # Calculate timestamp for this audio chunk
                                            # Timestamp = accumulated duration of all previous audio
                                            chunk_timestamp = self.accumulated_audio_duration
                                            chunk_duration = len(sub_chunk_48k) / 48000.0  # Duration in seconds

                                            # Add to queue (FIFO order) with timestamp - now at 48kHz!
                                            self.audio_chunks_queue.append((sub_chunk_48k, chunk_timestamp))

                                            # Update accumulated duration
                                            self.accumulated_audio_duration += chunk_duration

                                            # Log every chunk storage with RMS and timestamp
                                            if self.audio_chunk_count <= 5 or (self.audio_chunk_count % 10 == 0 and i == 0):
                                                logger.warning(f"📥 STORE: queue_idx={self.chunks_stored}, timestamp={chunk_timestamp:.3f}s, 48kHz samples={len(sub_chunk_48k)}, RMS={rms:.4f}")

                                            self.chunks_stored += 1
                                    else:
                                        logger.warning(f"Not enough 48kHz audio buffered yet ({len(self.audio_buffer_48k)} < {samples_48k_needed})")

                                    logger.info(f"Stored audio chunk #{self.audio_chunk_count} at queue indices {start_queue_idx}-{self.chunks_stored-1} (queue size: {len(self.audio_chunks_queue)})")

                                    # Log storage
                                    if self.audio_chunk_count % 5 == 0:
                                        logger.info(f"Audio chunk #{self.audio_chunk_count} sent to SDK. Queue size: {len(self.audio_chunks_queue)}, total stored: {self.chunks_stored}")

                        except Exception as e:
                            # MediaStreamError is expected when audio track ends (e.g., recording finished)
                            if "MediaStreamError" in str(type(e).__name__):
                                logger.info(f"Audio track ended (recording finished)")
                            else:
                                logger.error(f"Error processing audio frame: {e}", exc_info=True)
                            break

            # Add video track
            self.pc.addTrack(self.video_track)

            # Add buffered audio track (already created earlier, will output audio synchronized with video)
            self.audio_transceiver = self.pc.addTrack(self.buffered_audio_track)
            logger.info("Added buffered audio track to peer connection (will sync with video)")

            # Update video track's reference to buffered audio track
            self.video_track.buffered_audio_track = self.buffered_audio_track

            # Setup data channel for control (optional)
            channel = self.pc.createDataChannel("control")

            @channel.on("message")
            def on_message(message):
                logger.info(f"Data channel message: {message}")

            self.connected = True

            await self.send_message({
                "type": "ready",
                "message": "Server ready to receive offer"
            })

        except Exception as e:
            logger.error(f"Error in handle_connect: {e}", exc_info=True)
            await self.send_error(str(e))

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

            logger.info("Creating answer (with buffered audio track for synchronized audio-video)")

            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            # Send answer to client
            await self.send_message({
                "type": "answer",
                "sdp": self.pc.localDescription.sdp
            })

            logger.info("WebRTC handshake complete")

        except Exception as e:
            logger.error(f"Error in handle_offer: {e}", exc_info=True)
            await self.send_error(str(e))

    async def handle_answer(self, message: dict):
        """Handle WebRTC answer from client (for renegotiation)."""
        if not self.pc:
            return

        try:
            sdp = message.get("sdp")
            if not sdp:
                await self.send_error("Missing 'sdp' field in answer")
                return

            # Set remote description
            answer = RTCSessionDescription(sdp=sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

            logger.info("Renegotiation answer received and set")

        except Exception as e:
            logger.error(f"Error in handle_answer: {e}", exc_info=True)
            await self.send_error(str(e))

    async def handle_ice_candidate(self, message: dict):
        """Handle ICE candidate from client."""
        if not self.pc:
            return

        try:
            candidate_dict = message.get("candidate")
            if not candidate_dict:
                return

            # aiortc doesn't have from_sdp, we need to parse it manually
            # For now, skip adding client ICE candidates since peer reflexive discovery works
            # The connection completes successfully anyway
            logger.debug(f"Received ICE candidate (using peer reflexive discovery)")

        except Exception as e:
            logger.error(f"Error handling ICE candidate: {e}", exc_info=True)


    async def send_message(self, message: dict):
        """Send JSON message to client."""
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def send_error(self, error_message: str):
        """Send error message to client."""
        await self.send_message({
            "type": "error",
            "message": error_message
        })

    async def cleanup(self):
        """Cleanup resources."""
        if self.sdk:
            try:
                # Only close if SDK was fully initialized
                if hasattr(self.sdk, 'audio2motion_queue'):
                    self.sdk.close()
            except Exception as e:
                logger.error(f"Error closing SDK: {e}")

        if self.pc:
            await self.pc.close()

        logger.info("Session cleaned up")


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
        self.sessions: Dict[WebSocketServerProtocol, DittoWebRTCSession] = {}

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        logger.info(f"New client connected: {websocket.remote_address}")

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
                    elif msg_type == "answer":
                        await session.handle_answer(data)
                    elif msg_type == "ice-candidate":
                        await session.handle_ice_candidate(data)
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    await session.send_error("Invalid JSON")
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
                    await session.send_error(str(e))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            # Cleanup session
            await session.cleanup()
            del self.sessions[websocket]

    async def run(self):
        """Start the signaling server."""
        logger.info(f"Starting Ditto WebRTC Signaling Server on {self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Server running. Connect clients to ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ditto WebRTC Signaling Server")
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
        logger.info("Server stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
