"""
Ditto Avatar Module - Clean Interface for Conversational Systems

This module provides a clean interface for integrating Ditto avatar generation
into conversational AI systems like Gemini.

Usage:
    # Create avatar
    avatar = DittoAvatarModule(cfg_pkl, data_root, source_path)
    await avatar.initialize()

    # Feed audio from TTS
    avatar.process_audio(audio_chunk_16khz)

    # Get synchronized video + audio output
    video_frame, audio_chunk, timestamp = await avatar.get_next_frame()
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stream_pipeline_online import StreamSDK

logger = logging.getLogger(__name__)


class DittoAvatarModule:
    """
    Clean interface for Ditto avatar generation in conversational systems.

    This module handles:
    1. Audio buffering and processing
    2. Video frame generation
    3. Audio-video synchronization
    4. Output queue management

    Design:
        Input: 16kHz audio chunks (from TTS or passthrough)
        Output: Synchronized (video_frame, audio_chunk, timestamp) tuples
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        target_fps: int = 25,
        **kwargs
    ):
        """
        Initialize Ditto Avatar Module.

        Args:
            cfg_pkl: Path to Ditto configuration pickle
            data_root: Path to model data root
            source_path: Path to avatar source image/video
            target_fps: Target frames per second (default: 25)
            **kwargs: Additional Ditto configuration (max_size, emo, etc.)
        """
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.source_path = source_path
        self.target_fps = target_fps
        self.ditto_kwargs = kwargs

        # Ditto SDK
        self.sdk: Optional[StreamSDK] = None

        # Audio parameters
        self.model_rate = 16000  # Ditto input rate
        self.output_rate = 48000  # WebRTC output rate
        self.model_chunk_size = 6400  # 400ms @ 16kHz

        # Audio buffer for model
        self.audio_buffer = np.array([], dtype=np.float32)

        # Output queues
        self.video_queue = asyncio.Queue(maxsize=500)
        self.audio_queue = asyncio.Queue(maxsize=500)

        # Timing
        self._start_time = None
        self._frame_count = 0
        self._audio_chunk_count = 0

        # Statistics
        self._frames_generated = 0
        self._chunks_processed = 0

    async def initialize(self):
        """Initialize the Ditto SDK and setup frame callback."""
        logger.info(f"🎭 Initializing Ditto Avatar Module")
        logger.info(f"   Source: {self.source_path}")
        logger.info(f"   Target FPS: {self.target_fps}")

        # Create SDK
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root, **self.ditto_kwargs)

        # Setup SDK
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

        logger.info("✅ Ditto Avatar Module initialized")

    def _on_frame_generated(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """
        Callback when Ditto generates a video frame.

        This is called by the Ditto SDK when a new frame is ready.
        """
        self._frames_generated += 1

        try:
            # Queue video frame with index and timestamp
            self.video_queue.put_nowait((frame_rgb, frame_idx, timestamp))

            if self._frames_generated == 1:
                logger.info(f"🎬 First frame generated: idx={frame_idx}, ts={timestamp:.3f}s")

            if self._frames_generated % 100 == 0:
                logger.debug(f"📊 Generated {self._frames_generated} frames")

        except asyncio.QueueFull:
            logger.warning(f"⚠️ Video queue full, frame {frame_idx} dropped")

    def process_audio(
        self,
        audio_chunk: np.ndarray,
        upsample_for_output: bool = True
    ):
        """
        Process audio chunk for Ditto model and optionally queue for output.

        Args:
            audio_chunk: Audio chunk at 16kHz (numpy array, float32, mono)
            upsample_for_output: If True, upsample to 48kHz and queue for playback

        This method:
        1. Buffers audio until we have 6400 samples (400ms)
        2. Feeds complete chunks to Ditto model
        3. Optionally upsamples and queues for synchronized playback
        """
        # Validate input
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Process complete chunks
        while len(self.audio_buffer) >= self.model_chunk_size:
            # Extract chunk
            chunk = self.audio_buffer[:self.model_chunk_size]
            self.audio_buffer = self.audio_buffer[self.model_chunk_size:]

            # Feed to Ditto model
            if self.sdk:
                self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))
                self._chunks_processed += 1

                if self._chunks_processed == 1:
                    logger.info(f"🎤 First audio chunk processed: {len(chunk)} samples @ 16kHz")

            # Optionally queue upsampled audio for output
            if upsample_for_output:
                # Upsample 16kHz → 48kHz (3x)
                import scipy.signal
                chunk_48k = scipy.signal.resample_poly(chunk, up=3, down=1)

                # Split into 40ms sub-chunks (1920 samples @ 48kHz)
                # 6400 samples @ 16kHz = 400ms = 19200 samples @ 48kHz = 10 chunks
                samples_per_subchunk = 1920
                for i in range(0, len(chunk_48k), samples_per_subchunk):
                    sub_chunk = chunk_48k[i:i + samples_per_subchunk]

                    # Pad if needed
                    if len(sub_chunk) < samples_per_subchunk:
                        sub_chunk = np.pad(sub_chunk, (0, samples_per_subchunk - len(sub_chunk)))

                    try:
                        # Queue audio chunk
                        self.audio_queue.put_nowait(sub_chunk)
                        self._audio_chunk_count += 1
                    except asyncio.QueueFull:
                        logger.warning(f"⚠️ Audio queue full, chunk dropped")

    async def get_next_frame(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get next synchronized video frame and audio chunk.

        Returns:
            (video_rgb, audio_48k, timestamp)

        This method waits for both video and audio to be available,
        ensuring synchronized output.
        """
        # Get video frame
        video_rgb, frame_idx, video_timestamp = await self.video_queue.get()

        # Get corresponding audio chunk (FIFO)
        try:
            audio_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout waiting for audio, using silence")
            audio_chunk = np.zeros(1920, dtype=np.float32)

        self._frame_count += 1

        if self._frame_count % 100 == 0:
            logger.debug(f"📤 Output frame #{self._frame_count}: "
                        f"video_queue={self.video_queue.qsize()}, "
                        f"audio_queue={self.audio_queue.qsize()}")

        return (video_rgb, audio_chunk, video_timestamp)

    async def get_next_frame_nowait(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Non-blocking version of get_next_frame().

        Returns:
            (video_rgb, audio_48k, timestamp) or None if not available
        """
        try:
            video_rgb, frame_idx, video_timestamp = self.video_queue.get_nowait()
            audio_chunk = self.audio_queue.get_nowait()
            return (video_rgb, audio_chunk, video_timestamp)
        except asyncio.QueueEmpty:
            return None

    def get_queue_sizes(self) -> Tuple[int, int]:
        """
        Get current queue sizes.

        Returns:
            (video_queue_size, audio_queue_size)
        """
        return (self.video_queue.qsize(), self.audio_queue.qsize())

    def is_ready(self) -> bool:
        """Check if both video and audio are available."""
        return not self.video_queue.empty() and not self.audio_queue.empty()

    def close(self):
        """Close the Ditto SDK and cleanup resources."""
        if self.sdk:
            try:
                if hasattr(self.sdk, 'audio2motion_queue'):
                    self.sdk.close()
                logger.info("✅ Ditto SDK closed")
            except Exception as e:
                logger.error(f"❌ Error closing SDK: {e}")

    def get_statistics(self) -> dict:
        """
        Get module statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "frames_generated": self._frames_generated,
            "frames_output": self._frame_count,
            "chunks_processed": self._chunks_processed,
            "audio_chunks_queued": self._audio_chunk_count,
            "video_queue_size": self.video_queue.qsize(),
            "audio_queue_size": self.audio_queue.qsize(),
            "target_fps": self.target_fps,
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of how to use DittoAvatarModule in a conversational system."""

    # Initialize avatar
    avatar = DittoAvatarModule(
        cfg_pkl="outputs/cfg_f_model.pkl",
        data_root="./",
        source_path="examples/avatar.jpg",
        max_size=1920,
        emo=4  # neutral
    )

    await avatar.initialize()

    # Simulate audio input (e.g., from Gemini TTS)
    # In real usage, this would come from your TTS system
    async def simulate_audio_input():
        """Simulate continuous audio input."""
        while True:
            # Generate fake audio chunk (1 second @ 16kHz)
            audio_chunk = np.random.randn(16000).astype(np.float32) * 0.1

            # Process audio (this feeds Ditto and queues output)
            avatar.process_audio(audio_chunk, upsample_for_output=True)

            await asyncio.sleep(1.0)

    # Simulate video output (e.g., to WebRTC)
    async def consume_output():
        """Consume synchronized video + audio output."""
        while True:
            # Get next frame (blocking)
            video_frame, audio_chunk, timestamp = await avatar.get_next_frame()

            # Send to WebRTC or save to file
            print(f"Got frame: {video_frame.shape}, audio: {audio_chunk.shape}, ts: {timestamp:.3f}s")

            # In real usage, you would send to WebRTC tracks here

    # Run both tasks
    try:
        await asyncio.gather(
            simulate_audio_input(),
            consume_output()
        )
    except KeyboardInterrupt:
        print("\nStopping...")
        avatar.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
