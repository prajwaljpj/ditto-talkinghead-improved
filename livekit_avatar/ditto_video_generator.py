import sys
import os
import asyncio
import logging
import numpy as np
import cv2
import time
from typing import Optional, Union
from collections.abc import AsyncGenerator, AsyncIterator
from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    VideoGenerator,
)

# Configure logging
logger = logging.getLogger(__name__)

# Adjust system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from stream_pipeline_online import StreamSDK
except ImportError:
    raise ImportError("StreamSDK not found.")


def rgb_to_i420(frame_rgb: np.ndarray, width: int, height: int) -> bytes:
    """Convert RGB to I420."""
    if frame_rgb.shape[0] != height or frame_rgb.shape[1] != width:
        frame_rgb = cv2.resize(
            frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR
        )

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    yuv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv_frame.tobytes()


class DittoVideoGenerator(VideoGenerator):
    """
    Ditto video generator with LiveKit audio_wave pattern.

    Correctly extracts audio from buffer position matching video representation.
    """

    def __init__(
        self,
        options: AvatarOptions,
        data_root: str,
        cfg_pkl: str,
        source_path: str,
    ):
        self._options = options
        self._audio_queue = asyncio.Queue[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._audio_resampler: Optional[rtc.AudioResampler] = None
        self._loop = asyncio.get_event_loop()

        # Ditto configuration
        self.chunksize = (3, 5, 2)  # (past, current, future)
        self.split_len = 6480  # Total chunk size
        self.stride = 3200  # Advance by 5 frames

        # Video frames from Ditto callback
        self._video_frames: list[rtc.VideoFrame] = []

        # Tracking
        self._chunks_processed = 0
        self._frames_yielded = 0

        # Initialize Ditto
        logger.info("Initializing Ditto StreamSDK...")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.sdk.setup(
            source_path,
            output_path="/dev/null",
            frame_callback=self._on_video_frame,
            online_mode=True,
            fps=options.video_fps,
        )
        self.sdk.setup_Nd(N_d=1000000)

        # Warmup
        logger.info("Warming up Ditto...")
        silent_chunk = np.zeros(self.split_len, dtype=np.float32)
        for i in range(2):
            self.sdk.run_chunk(silent_chunk, self.chunksize)
        self._video_frames.clear()
        logger.info("✅ Ditto ready")

        # AudioByteStream for frame-aligned chunks
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=640,
        )

        # Buffer for accumulating audio samples for Ditto chunks
        self._sample_buffer = np.zeros((0,), dtype=np.float32)

        # State machine for clean idle ↔ TTS transitions
        self._is_speaking = False  # False = idle, True = TTS/speaking
        self._pending_transition = False  # Flag for state change

        # Add global padding (from inference.py line 48)
        padding = np.zeros(self.chunksize[0] * 640, dtype=np.float32)
        self._sample_buffer = np.concatenate([padding, self._sample_buffer])

        self._chunks_processed = 0
        self._frames_yielded = 0

    def _on_video_frame(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """Video frame callback from Ditto."""
        try:
            i420_data = rgb_to_i420(
                frame_rgb, self._options.video_width, self._options.video_height
            )
            video_frame = rtc.VideoFrame(
                data=i420_data,
                width=self._options.video_width,
                height=self._options.video_height,
                type=rtc.VideoBufferType.I420,
            )
            self._video_frames.append(video_frame)
        except Exception as e:
            logger.error(f"Video callback error: {e}", exc_info=True)

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Push audio frame from agent."""
        if isinstance(frame, AudioSegmentEnd):
            logger.info("📨 Received AudioSegmentEnd from agent")
        else:
            logger.info(
                f"=================Audio Frame: {frame}=============,"
                f"📨 Received audio frame: {frame.samples_per_channel} samples, "
                f"{frame.sample_rate}Hz, {frame.num_channels}ch"
            )
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        """
        Called by AvatarRunner on interruption.

        For better conversational experience, we DON'T actually clear buffers.
        """
        logger.info(
            "🔄 Interruption detected - allowing avatar to complete current speech"
        )
        # For better UX, don't clear - but if we did clear:
        # self._audio_output_buffer = np.zeros((0,), dtype=np.float32)

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._video_generation_impl()

    async def _video_generation_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """Main generation loop."""
        while True:
            try:
                # Wait for audio with timeout
                timeout = 0.5 / self._options.video_fps  # ~20ms for 25fps
                frame = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # No TTS - generate idle frame
                async for output in self._process_idle():
                    yield output
                continue

            # Handle AudioSegmentEnd separately (doesn't have audio data)
            if isinstance(frame, AudioSegmentEnd):
                # Flush AudioByteStream
                audio_frames = self._audio_bstream.flush()
            else:
                # Resample if needed (only for actual audio frames)
                resampled_frames = self._resample_to_16k(frame)

                # Chunk into frame-aligned pieces
                audio_frames = []
                for rf in resampled_frames:
                    for synced in self._audio_bstream.push(rf.data):
                        audio_frames.append(synced)

            # Generate video for each audio frame
            for audio_frame in audio_frames:
                async for output in self._process_audio_frame(audio_frame):
                    yield output

            # Yield AudioSegmentEnd and transition back to idle
            if isinstance(frame, AudioSegmentEnd):
                # Flush remaining TTS
                async for output in self._flush():
                    yield output

                # Transition: Speaking → Idle
                logger.info("🔇 Transition: Speaking → Idle")
                self._is_speaking = False

                # Yield AudioSegmentEnd to notify runner
                yield AudioSegmentEnd()

    async def _process_audio_frame(
        self, audio_frame: rtc.AudioFrame
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame]:
        """Process one frame-aligned audio chunk."""

        # Check if this is real TTS (transition from idle → speaking)
        # Idle frames are silent, TTS frames have energy
        audio_int16 = np.frombuffer(audio_frame.data, dtype=np.int16)
        audio_energy = np.abs(audio_int16).mean()
        is_tts = audio_energy > 10  # Threshold to detect non-silent audio

        # Transition: idle → TTS (clear contaminated buffers!)
        if not self._is_speaking and is_tts:
            logger.info("🎤 Transition: Idle → Speaking (clearing buffers)")
            self._is_speaking = True

            # Clear sample buffer (remove idle contamination)
            self._sample_buffer = np.zeros((0,), dtype=np.float32)

            # Clear video frames (remove any pending idle frames)
            # This prevents idle video from being paired with TTS audio
            self._video_frames.clear()

            # Add padding for past context (represents "at rest" before speech)
            # Without this, first TTS chunk has no context about previous mouth position
            padding = np.zeros(
                self.chunksize[0] * 640, dtype=np.float32
            )  # 1920 samples
            self._sample_buffer = np.concatenate([padding, self._sample_buffer])

            # Reset chunks counter (fresh start)
            self._chunks_processed = 0

            logger.info("✓ Buffers cleared with padding, ready for TTS")

        # Convert to float32 and add to buffer
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        self._sample_buffer = np.concatenate([self._sample_buffer, audio_float32])

        # Process when buffer has enough samples
        async for output in self._process_buffer():
            yield output

    async def _process_idle(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame]:
        """
        Generate idle frame with silent audio for Ditto processing.

        Note: We yield ONLY video during idle, not audio.
        This allows AvatarRunner to correctly track _audio_playing state
        (False during idle, True during TTS).
        """
        # Add silent samples to buffer for Ditto processing
        silent = np.zeros(640, dtype=np.float32)
        self._sample_buffer = np.concatenate([self._sample_buffer, silent])

        # Process if buffer ready - but ONLY yield video frames
        async for output in self._process_buffer():
            if isinstance(output, rtc.VideoFrame):
                yield output  # Yield idle video
            # Skip audio frames during idle (no silent audio yielded)

    async def _process_buffer(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame]:
        """Process buffer when ready (>= 6480 samples)."""
        while len(self._sample_buffer) >= self.split_len:
            # Extract chunk
            chunk = self._sample_buffer[: self.split_len].copy()
            logger.info(
                f"===========Chunk being processed: {chunk}================\n===============Chunk Shape: {chunk.shape}===================="
            )
            # Determine which audio samples will be output
            # CRITICAL: Ditto handles lookahead INTERNALLY in its model
            # We should NOT add lookahead offset when extracting audio
            # Just skip padding on first chunk, extract stride portion for output

            if self._chunks_processed == 0:
                # First chunk: skip padding only (1920 samples)
                # Video frames 0-4 represent TTS audio samples 0-3199
                audio_start = self.chunksize[0] * 640  # 1920 (padding)
                audio_end = audio_start + self.stride  # 1920 + 3200 = 5120
                audio_for_frames = chunk[audio_start:audio_end].copy()
                logger.debug(
                    f"Chunk 0: Extracting audio[{audio_start}:{audio_end}] (skipping padding)"
                )
            else:
                # Subsequent chunks: extract first stride samples
                # These represent the "new" audio advanced by stride
                audio_start = 0
                audio_end = self.stride  # 3200
                audio_for_frames = chunk[audio_start:audio_end].copy()
                logger.debug(
                    f"Chunk {self._chunks_processed}: Extracting audio[{audio_start}:{audio_end}]"
                )

            # Process through Ditto
            await self._loop.run_in_executor(
                None, self.sdk.run_chunk, chunk, self.chunksize
            )
            self._chunks_processed += 1

            # Advance by stride (5 frames = 3200 samples)
            self._sample_buffer = self._sample_buffer[self.stride :]

            # Ditto outputs 5 video frames
            # Create 5 audio frames from the processed samples
            num_video = len(self._video_frames)

            for i in range(min(num_video, 5)):
                video = self._video_frames.pop(0)

                # Extract 640 samples for this frame
                start = i * 640
                end = start + 640
                audio_samples = audio_for_frames[start:end]

                # Create audio frame from samples
                audio_int16 = (audio_samples * 32768.0).astype(np.int16)
                audio_int16 = np.clip(audio_int16, -32768, 32767)
                audio_frame = rtc.AudioFrame(
                    data=audio_int16.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=640,
                )

                # Yield video and audio immediately (no delay)
                yield video
                yield audio_frame
                self._frames_yielded += 1

                if self._frames_yielded % 25 == 0:
                    logger.info(
                        f"Frames: {self._frames_yielded}, Chunks: {self._chunks_processed}"
                    )

    async def _flush(self) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame]:
        """Flush remaining buffer."""
        if len(self._sample_buffer) > 0:
            # Pad to complete chunk
            if len(self._sample_buffer) < self.split_len:
                padding = self.split_len - len(self._sample_buffer)
                self._sample_buffer = np.concatenate(
                    [self._sample_buffer, np.zeros(padding, dtype=np.float32)]
                )

            async for output in self._process_buffer():
                yield output

    def _resample_to_16k(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        """Resample to 16kHz mono."""
        if frame.sample_rate != 16000 or frame.num_channels != 1:
            if self._audio_resampler is None:
                self._audio_resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=16000,
                    num_channels=1,
                )
            return list(self._audio_resampler.push(frame))
        return [frame]

    async def aclose(self) -> None:
        """Cleanup method for compatibility with AvatarRunner."""
        logger.info("Closing DittoVideoGenerator...")
        # Add any cleanup logic here if needed (close files, release resources, etc.)
        pass
