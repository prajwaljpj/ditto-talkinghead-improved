"""
LiveKit Ditto Avatar Agent

This agent uses LiveKit to handle ALL WebRTC complexity, allowing Python
to focus on what it does best: ML inference.

Architecture:
    Browser → LiveKit Cloud → This Agent
    - Browser sends audio via LiveKit
    - Agent generates video with Ditto
    - Agent sends video via LiveKit
    - LiveKit handles all WebRTC/NAT/ICE/codecs

Usage:
    # Local LiveKit server (for development):
    docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
        -e LIVEKIT_KEYS="devkey: devse

cret" \
        livekit/livekit-server:latest

    # Run agent:
    python livekit_ditto_agent.py \
        --cfg_pkl <path> \
        --data_root <path> \
        --source <avatar.jpg>
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# LiveKit
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stream_pipeline_online import StreamSDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DittoAvatarAgent:
    """
    LiveKit agent that generates Ditto avatar video from audio input.

    This agent:
    1. Subscribes to audio from participant
    2. Processes audio through Ditto
    3. Publishes generated video frames
    4. Uses LiveKit for ALL WebRTC handling

    No more threading issues, no more async/queue problems!
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        **ditto_kwargs
    ):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.source_path = source_path
        self.ditto_kwargs = ditto_kwargs

        # Ditto SDK
        self.sdk: Optional[StreamSDK] = None

        # LiveKit components
        self.room: Optional[rtc.Room] = None
        self.video_source: Optional[rtc.VideoSource] = None
        self.audio_source: Optional[rtc.AudioSource] = None

        # Audio processing
        self.audio_buffer = np.array([], dtype=np.float32)
        self.model_chunk_size = 6400  # 400ms @ 16kHz

        # Output audio buffer (for passthrough/echo)
        self.output_audio_buffer = []
        self.accumulated_audio_time = 0.0  # For timestamps

        # Frame timing for smooth playback
        self.target_fps = 25  # Match Ditto output
        self.last_frame_time = None
        self.frame_interval = 1.0 / self.target_fps

        # Statistics
        self._frames_generated = 0
        self._audio_chunks_processed = 0
        self._frames_dropped = 0

    async def initialize(self):
        """Initialize Ditto SDK."""
        logger.info("🎭 Initializing Ditto SDK...")

        # Initialize SDK (blocking operation, but only done once)
        await asyncio.to_thread(self._init_sdk_sync)

        logger.info("✅ Ditto SDK initialized")

    def _init_sdk_sync(self):
        """Synchronous SDK initialization (run in thread pool)."""
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root, **self.ditto_kwargs)

        setup_kwargs = {
            "online_mode": True,
            "N_d": -1,
        }
        setup_kwargs.update(self.ditto_kwargs)

        self.sdk.setup(
            self.source_path,
            output_path=None,
            frame_callback=self._on_frame_generated,
            **setup_kwargs
        )

    def _on_frame_generated(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """
        Callback from Ditto SDK when frame is ready.

        NOTE: This is called from Ditto's worker thread!
        But now we don't need complex queue coordination - we just
        schedule the frame to be sent via LiveKit.
        """
        self._frames_generated += 1

        if self._frames_generated == 1:
            logger.info(f"🎬 First frame generated: {frame_rgb.shape}")
            self.last_frame_time = time.time()

        # Frame pacing to reduce jitter
        current_time = time.time()
        if self.last_frame_time:
            elapsed = current_time - self.last_frame_time
            # Drop frames if we're getting ahead (reduces jitter)
            if elapsed < self.frame_interval * 0.8:
                self._frames_dropped += 1
                if self._frames_dropped % 10 == 0:
                    logger.debug(f"⏩ Dropped {self._frames_dropped} frames (pacing)")
                return

            # Log if we're falling behind
            if elapsed > self.frame_interval * 1.5:
                logger.warning(f"⚠️ Frame delay: {elapsed*1000:.1f}ms (target: {self.frame_interval*1000:.1f}ms)")

        self.last_frame_time = current_time

        # Schedule frame to be sent (thread-safe)
        if self.video_source:
            try:
                # Create video frame
                video_frame = rtc.VideoFrame(
                    width=frame_rgb.shape[1],
                    height=frame_rgb.shape[0],
                    type=rtc.VideoBufferType.RGBA,
                    data=self._rgb_to_rgba(frame_rgb)
                )

                # Capture frame (thread-safe operation)
                self.video_source.capture_frame(video_frame)

                if self._frames_generated % 100 == 0:
                    logger.info(f"📊 Sent {self._frames_generated} frames ({self._frames_dropped} dropped for pacing)")

            except Exception as e:
                logger.error(f"Error sending frame: {e}")

    def _rgb_to_rgba(self, rgb: np.ndarray) -> bytes:
        """Convert RGB to RGBA (add alpha channel)."""
        h, w, _ = rgb.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255  # Full opacity
        return rgba.tobytes()

    async def process_audio_frame(self, audio_frame: rtc.AudioFrame):
        """
        Process incoming audio frame from participant.

        This is called from the main async loop, so no threading issues!
        """
        # Convert to numpy (mono, float32)
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Handle stereo → mono if needed
        if audio_frame.num_channels == 2:
            # Interleaved stereo: [L1, R1, L2, R2, ...]
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

        # Resample if needed (LiveKit typically uses 48kHz, Ditto needs 16kHz)
        if audio_frame.sample_rate == 48000:
            import scipy.signal
            audio_data = scipy.signal.resample_poly(audio_data, up=1, down=3)
        elif audio_frame.sample_rate != 16000:
            # Generic resampling
            import scipy.signal
            num_samples = int(len(audio_data) * 16000 / audio_frame.sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)

        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

        # Process complete chunks
        while len(self.audio_buffer) >= self.model_chunk_size:
            chunk = self.audio_buffer[:self.model_chunk_size]
            self.audio_buffer = self.audio_buffer[self.model_chunk_size:]

            # Feed to Ditto (run in thread pool to avoid blocking)
            await asyncio.to_thread(
                self.sdk.run_chunk,
                chunk,
                (3, 5, 2)  # chunksize
            )

            self._audio_chunks_processed += 1

            if self._audio_chunks_processed % 10 == 0:
                logger.debug(f"📤 Processed {self._audio_chunks_processed} audio chunks")

            # Store audio for passthrough/echo (optional)
            # You can send this back if you want the user to hear themselves
            self.output_audio_buffer.append(chunk)

    def close(self):
        """Cleanup resources."""
        if self.sdk:
            try:
                if hasattr(self.sdk, 'audio2motion_queue'):
                    self.sdk.close()
                logger.info("✅ Ditto SDK closed")
            except Exception as e:
                logger.error(f"Error closing SDK: {e}")


async def entrypoint(ctx: JobContext):
    """
    LiveKit agent entrypoint.

    This function is called when a participant joins the room.

    Configuration is read from environment variables:
    - DITTO_CFG_PKL
    - DITTO_DATA_ROOT
    - DITTO_SOURCE
    - DITTO_MAX_SIZE (optional, default: 1920)
    - DITTO_EMO (optional, default: 4)
    """
    import os

    logger.info(f"🚀 Agent starting for room: {ctx.room.name}")

    # Read configuration from environment
    cfg_pkl = os.getenv("DITTO_CFG_PKL", "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    data_root = os.getenv("DITTO_DATA_ROOT", "checkpoints/ditto_trt_custom2/")
    source_path = os.getenv("DITTO_SOURCE", "avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg")
    max_size = int(os.getenv("DITTO_MAX_SIZE", "1920"))
    emo = int(os.getenv("DITTO_EMO", "4"))

    logger.info(f"📋 Configuration:")
    logger.info(f"   cfg_pkl: {cfg_pkl}")
    logger.info(f"   data_root: {data_root}")
    logger.info(f"   source: {source_path}")
    logger.info(f"   max_size: {max_size}")
    logger.info(f"   emo: {emo}")

    # Create Ditto agent
    agent = DittoAvatarAgent(
        cfg_pkl=cfg_pkl,
        data_root=data_root,
        source_path=source_path,
        max_size=max_size,
        emo=emo
    )

    # Initialize Ditto
    await agent.initialize()

    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.room = ctx.room

    # Create video source (720p @ 25fps)
    agent.video_source = rtc.VideoSource(1280, 720)

    # Publish video track
    video_track = rtc.LocalVideoTrack.create_video_track("ditto_avatar", agent.video_source)
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(video_track, video_options)

    logger.info("✅ Video track published to room")

    # Subscribe to audio from first participant
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"📡 Subscribed to track: {track.kind} from {participant.identity}")

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("🎤 Processing audio from participant")

            audio_stream = rtc.AudioStream(track)

            async def process_audio_stream():
                async for audio_frame_event in audio_stream:
                    await agent.process_audio_frame(audio_frame_event.frame)

            # Start processing in background
            asyncio.create_task(process_audio_stream())

    logger.info("✅ Agent ready and waiting for participants")

    # Keep agent alive
    try:
        await asyncio.Future()  # Run forever
    finally:
        agent.close()


async def request_fnc(ctx: JobContext):
    """
    Request handler - called before entrypoint to validate job.

    You can use this to reject jobs or set metadata.
    """
    logger.info(f"📩 Job request for room: {ctx.room.name}")
    await ctx.accept()


if __name__ == "__main__":
    """
    Start the LiveKit agent.

    Configuration via environment variables:

    LiveKit (required):
    - LIVEKIT_URL: ws://localhost:7880 or wss://your-project.livekit.cloud
    - LIVEKIT_API_KEY: your-api-key
    - LIVEKIT_API_SECRET: your-api-secret

    Ditto (optional, with defaults):
    - DITTO_CFG_PKL: path to config pickle
    - DITTO_DATA_ROOT: path to model data
    - DITTO_SOURCE: path to avatar image
    - DITTO_MAX_SIZE: max image dimension (default: 1920)
    - DITTO_EMO: emotion 0-7 (default: 4=neutral)
    """
    # Run LiveKit agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
        )
    )
