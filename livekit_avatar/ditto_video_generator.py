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

# Adjust system path to import stream_pipeline_online from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from stream_pipeline_online import StreamSDK
except ImportError:
    raise ImportError(
        "StreamSDK not found. Ensure 'stream_pipeline_online.py' is in the project root folder."
    )


def rgb_to_i420(frame_rgb: np.ndarray, width: int, height: int) -> bytes:
    """Converts an RGB NumPy array to I420 byte data, resizing if necessary."""
    if frame_rgb.shape[0] != height or frame_rgb.shape[1] != width:
        frame_rgb = cv2.resize(
            frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR
        )

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    yuv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv_frame.tobytes()


class DittoVideoGenerator(VideoGenerator):
    """
    VideoGenerator implementation that uses Ditto StreamSDK for avatar animation.

    This generator:
    - Receives audio frames from AvatarRunner via push_audio()
    - Feeds audio to Ditto SDK for lip-sync video generation
    - Yields synchronized audio and video frames
    - Generates silent audio/idle video when no TTS audio is available

    The audio-driven approach naturally handles states:
    - Real audio → Speaking animation with lip-sync
    - Silent audio → Idle animation (breathing, blinking)
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

        # Ditto configuration
        self.data_root = data_root
        self.cfg_pkl = cfg_pkl
        self.source_path = source_path

        # Ditto audio processing configuration
        self.chunksize = (3, 5, 2)  # Ditto-specific chunking
        self.split_len = 6480  # Samples per chunk for 16kHz audio
        self.ditto_sample_rate = 16000  # Ditto expects 16kHz
        self.silent_chunk = np.zeros(self.split_len, dtype=np.float32)

        # Audio buffering for Ditto's chunking requirements
        self._sdk_audio_buffer = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
        self._sdk_lock = asyncio.Lock()

        # AudioByteStream to chunk audio frames to match video frame rate
        # This ensures each audio frame corresponds to exactly one video frame
        samples_per_video_frame = options.audio_sample_rate // options.video_fps
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=samples_per_video_frame,
        )
        logger.info(f"Audio chunking: {samples_per_video_frame} samples per frame @ {options.video_fps} fps")

        # Video frame storage (populated by Ditto callback)
        self._video_frame_queue = asyncio.Queue[rtc.VideoFrame](maxsize=10)

        # Store event loop reference for thread-safe coroutine scheduling
        self._loop = asyncio.get_event_loop()

        # Initialize Ditto StreamSDK
        logger.info("Initializing Ditto StreamSDK...")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.sdk.setup(
            source_path,
            output_path="/dev/null",  # We use frame callback instead
            frame_callback=self._handle_generated_frame,
            online_mode=True,
            fps=options.video_fps,
        )
        self.sdk.setup_Nd(N_d=1000000)  # Large number for continuous generation
        logger.info("✅ Ditto StreamSDK initialized")

        # Warmup: Generate a few dummy frames to initialize CUDA/TensorRT
        logger.info("Warming up Ditto model...")
        self._warmup_model()
        logger.info("✅ Model warmup complete")

    def _handle_generated_frame(
        self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float
    ):
        """
        Callback from StreamSDK (runs in SDK's worker thread).
        Converts and queues video frames for the main generation loop.
        """
        logger.info(f"🎥 Callback received frame {frame_idx} at {timestamp:.3f}s")
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

            # Queue the video frame (non-blocking to avoid callback delays)
            try:
                self._video_frame_queue.put_nowait(video_frame)
                logger.info(f"  ✅ Frame {frame_idx} queued successfully (queue size: {self._video_frame_queue.qsize()})")
            except asyncio.QueueFull:
                logger.warning(f"  ⚠️ Video frame queue full, dropping frame {frame_idx}")

        except Exception as e:
            logger.error(f"  ❌ Error in callback for frame {frame_idx}: {e}", exc_info=True)

    def _warmup_model(self):
        """
        Warm up the Ditto model by generating a few dummy frames.
        This initializes CUDA, loads TensorRT engines, etc.
        """
        # Generate 3-5 warmup frames with silent audio
        warmup_audio = np.zeros(self.split_len, dtype=np.float32)
        for i in range(3):
            try:
                self.sdk.run_chunk(warmup_audio, self.chunksize)
            except Exception as e:
                logger.warning(f"Warmup frame {i} failed: {e}")

        # Clear the video frame queue after warmup
        while not self._video_frame_queue.empty():
            try:
                self._video_frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # -- VideoGenerator abstract methods --

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Called by AvatarRunner to push audio frames to the generator."""
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        """Called by AvatarRunner to clear the audio buffer on interruption."""
        logger.info("Clearing audio buffer (interruption)")
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset AudioByteStream
        self._audio_bstream.flush()

        # Reset audio buffer for Ditto
        self._sdk_audio_buffer = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """
        Generate a continuous stream of video and audio frames.

        Flow:
        1. Get audio from queue (TTS) or timeout (idle)
        2. Resample audio to 16kHz mono if needed
        3. Buffer audio to Ditto's chunk size (6480 samples)
        4. Feed chunks to Ditto SDK
        5. Yield audio frames for playback
        6. Yield video frames from Ditto
        7. Yield AudioSegmentEnd when segment complete
        """
        return self._video_generation_impl()

    # -- End of VideoGenerator abstract methods --

    async def _video_generation_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        """Main generation loop."""

        while True:
            # Try to get audio from queue (TTS output or AudioSegmentEnd)
            try:
                # Timeout shorter than frame interval to avoid starvation
                frame = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.5 / self._options.video_fps
                )
                self._audio_queue.task_done()
                logger.debug(f"📥 Received frame from queue: {type(frame).__name__}")
            except asyncio.TimeoutError:
                # No TTS audio available - enter continuous idle generation mode
                # Keep generating idle frames until TTS audio arrives
                logger.info("⏰ Queue timeout - entering continuous idle mode")
                try:
                    async for frame in self._generate_continuous_idle():
                        yield frame
                    logger.info("✅ Exiting idle mode (TTS audio available)")
                except Exception as e:
                    logger.error(f"❌ Error in idle generation: {e}", exc_info=True)
                continue

            # Handle AudioSegmentEnd (marks end of TTS segment)
            if isinstance(frame, AudioSegmentEnd):
                # Flush any remaining audio in buffer
                await self._flush_audio_buffer()

                # Yield remaining video frames
                while not self._video_frame_queue.empty():
                    try:
                        video_frame = self._video_frame_queue.get_nowait()
                        yield video_frame
                    except asyncio.QueueEmpty:
                        break

                # Signal segment end
                yield AudioSegmentEnd()
                continue

            # Resample if necessary (Ditto expects 16kHz mono)
            resampled_frames: list[rtc.AudioFrame] = []
            if (
                frame.sample_rate != self.ditto_sample_rate
                or frame.num_channels != 1
            ):
                if not self._audio_resampler:
                    self._audio_resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=self.ditto_sample_rate,
                        num_channels=1,
                    )

                for f in self._audio_resampler.push(frame):
                    resampled_frames.append(f)
            else:
                resampled_frames.append(frame)

            # Push resampled audio through AudioByteStream to chunk to video frame rate
            # This creates fixed-size audio chunks that match video frames (320 samples @ 50fps)
            synced_audio_frames: list[rtc.AudioFrame] = []
            for resampled_frame in resampled_frames:
                synced_audio_frames += self._audio_bstream.push(resampled_frame.data)

            # Process each synchronized audio frame
            for synced_audio_frame in synced_audio_frames:
                # Convert to float32 for Ditto
                audio_data_int16 = np.frombuffer(synced_audio_frame.data, dtype=np.int16)
                audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

                # Add to Ditto's buffer
                self._sdk_audio_buffer = np.concatenate([self._sdk_audio_buffer, audio_data_float])

                # Process Ditto buffer when we have enough samples
                # Note: We may need to accumulate multiple synced frames before feeding to Ditto
                if len(self._sdk_audio_buffer) >= self.split_len:
                    # Extract chunk for Ditto
                    ditto_chunk = self._sdk_audio_buffer[: self.split_len]
                    self._sdk_audio_buffer = self._sdk_audio_buffer[self.chunksize[1] * 640 :]

                    # Feed to Ditto SDK (generates ~20 video frames for this chunk)
                    logger.debug(f"Feeding {len(ditto_chunk)} samples to Ditto (queue: {self._video_frame_queue.qsize()})")
                    async with self._sdk_lock:
                        await self._loop.run_in_executor(
                            None, self.sdk.run_chunk, ditto_chunk, self.chunksize
                        )
                    logger.debug(f"Ditto processing complete (queue: {self._video_frame_queue.qsize()})")

                # Now yield this synced audio frame with corresponding video frame
                # This ensures 1:1 audio:video frame ratio for perfect sync
                yield synced_audio_frame

                # Yield corresponding video frame from Ditto
                try:
                    video_frame = await asyncio.wait_for(
                        self._video_frame_queue.get(),
                        timeout=0.5
                    )
                    yield video_frame
                except asyncio.TimeoutError:
                    logger.debug("Timeout waiting for video frame from Ditto")

    async def _generate_continuous_idle(self):
        """
        Continuously generate idle frames until TTS audio arrives.

        This keeps the video stream running smoothly during listening mode
        without gaps between idle generation cycles.
        """
        # Pre-fill buffer once at the start
        logger.info("🎬 Pre-filling buffer for continuous idle mode")
        async for frame in self._prefill_idle_buffer():
            yield frame

        frames_generated = 0
        while True:
            # Check if TTS audio has arrived (non-blocking)
            if not self._audio_queue.empty():
                logger.info(f"🎯 TTS audio detected, exiting idle mode (generated {frames_generated} idle frames)")
                return

            # Generate one idle frame
            silent_frame = self._create_silent_audio_frame()
            synced_audio_frames = self._audio_bstream.push(silent_frame.data)

            for synced_audio_frame in synced_audio_frames:
                # Convert to float32 for Ditto
                audio_data_int16 = np.frombuffer(synced_audio_frame.data, dtype=np.int16)
                audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

                # Add to Ditto's buffer
                self._sdk_audio_buffer = np.concatenate([self._sdk_audio_buffer, audio_data_float])

                # Process Ditto buffer when we have enough samples
                if len(self._sdk_audio_buffer) >= self.split_len:
                    ditto_chunk = self._sdk_audio_buffer[: self.split_len]
                    self._sdk_audio_buffer = self._sdk_audio_buffer[self.chunksize[1] * 640 :]

                    # Feed to Ditto (non-blocking)
                    async with self._sdk_lock:
                        await self._loop.run_in_executor(
                            None, self.sdk.run_chunk, ditto_chunk, self.chunksize
                        )

                # Yield audio frame
                yield synced_audio_frame
                frames_generated += 1

                # Yield video frame
                try:
                    video_frame = await asyncio.wait_for(
                        self._video_frame_queue.get(),
                        timeout=0.1
                    )
                    yield video_frame
                    frames_generated += 1
                except asyncio.TimeoutError:
                    logger.debug(f"⏰ Video frame timeout at idle frame {frames_generated}")

    async def _prefill_idle_buffer(self):
        """Pre-fill the audio buffer to trigger initial Ditto call."""
        samples_per_frame = self._options.audio_sample_rate // self._options.video_fps
        frames_to_prefill = (self.split_len // samples_per_frame) + 1

        logger.debug(f"  📦 Pre-filling with {frames_to_prefill} frames")

        buffered_frames = []
        for i in range(frames_to_prefill):
            silent_frame = self._create_silent_audio_frame()
            synced_audio_frames = self._audio_bstream.push(silent_frame.data)

            for synced_audio_frame in synced_audio_frames:
                audio_data_int16 = np.frombuffer(synced_audio_frame.data, dtype=np.int16)
                audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

                self._sdk_audio_buffer = np.concatenate([self._sdk_audio_buffer, audio_data_float])
                buffered_frames.append(synced_audio_frame)

                if len(self._sdk_audio_buffer) >= self.split_len:
                    ditto_chunk = self._sdk_audio_buffer[: self.split_len]
                    self._sdk_audio_buffer = self._sdk_audio_buffer[self.chunksize[1] * 640 :]

                    logger.debug(f"  🎨 Initial Ditto call (queue: {self._video_frame_queue.qsize()})")
                    async with self._sdk_lock:
                        await self._loop.run_in_executor(
                            None, self.sdk.run_chunk, ditto_chunk, self.chunksize
                        )

        # Wait for callbacks to populate queue
        await asyncio.sleep(0.05)
        logger.debug(f"  ✅ Pre-fill complete, queue has {self._video_frame_queue.qsize()} frames")

        # Yield buffered frames
        for audio_frame in buffered_frames:
            yield audio_frame
            try:
                video_frame = await asyncio.wait_for(
                    self._video_frame_queue.get(),
                    timeout=0.1
                )
                yield video_frame
            except asyncio.TimeoutError:
                logger.debug("  ⏰ Timeout during pre-fill yield")

    async def _generate_idle_frames(self):
        """
        Generate idle video frames by feeding silent audio to Ditto.

        During idle (user speaking/listening mode), we need to continuously
        generate video frames to show idle animations (breathing, blinking).

        Strategy: Pre-fill the audio buffer to trigger Ditto BEFORE yielding frames,
        so video frames are ready when we start yielding.
        """
        samples_per_frame = self._options.audio_sample_rate // self._options.video_fps
        num_frames_to_generate = (self.split_len // samples_per_frame) + 2

        logger.info(f"🎬 Generating {num_frames_to_generate} idle frames (no TTS audio available)")

        # PHASE 1: Pre-fill buffer to trigger initial Ditto call
        # This ensures video frames are ready before we start yielding
        frames_to_prefill = (self.split_len // samples_per_frame) + 1
        logger.debug(f"  📦 Pre-filling buffer with {frames_to_prefill} frames")

        buffered_audio_frames = []
        for i in range(frames_to_prefill):
            silent_frame = self._create_silent_audio_frame()
            synced_audio_frames = self._audio_bstream.push(silent_frame.data)

            for synced_audio_frame in synced_audio_frames:
                # Convert to float32 for Ditto
                audio_data_int16 = np.frombuffer(synced_audio_frame.data, dtype=np.int16)
                audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

                # Add to Ditto's buffer
                self._sdk_audio_buffer = np.concatenate([self._sdk_audio_buffer, audio_data_float])

                # Save for yielding later
                buffered_audio_frames.append(synced_audio_frame)

                # Process Ditto buffer when we have enough samples
                if len(self._sdk_audio_buffer) >= self.split_len:
                    ditto_chunk = self._sdk_audio_buffer[: self.split_len]
                    self._sdk_audio_buffer = self._sdk_audio_buffer[self.chunksize[1] * 640 :]

                    logger.info(f"  🎨 Feeding {len(ditto_chunk)} samples to Ditto (queue size before: {self._video_frame_queue.qsize()})")
                    start_time = time.time()
                    async with self._sdk_lock:
                        await self._loop.run_in_executor(
                            None, self.sdk.run_chunk, ditto_chunk, self.chunksize
                        )
                    elapsed = (time.time() - start_time) * 1000
                    logger.info(f"  ✅ Ditto processing complete in {elapsed:.1f}ms (queue size after: {self._video_frame_queue.qsize()})")

        # Give Ditto callbacks a moment to populate the queue
        await asyncio.sleep(0.05)

        logger.debug(f"  ✅ Pre-fill complete, video queue has {self._video_frame_queue.qsize()} frames")

        # PHASE 2: Yield the buffered audio frames with their corresponding video frames
        frames_yielded = 0
        for synced_audio_frame in buffered_audio_frames:
            # Yield audio frame
            yield synced_audio_frame
            frames_yielded += 1

            # Yield corresponding video frame
            try:
                video_frame = await asyncio.wait_for(
                    self._video_frame_queue.get(),
                    timeout=0.1  # Shorter timeout since frames should be ready
                )
                yield video_frame
                frames_yielded += 1
            except asyncio.TimeoutError:
                logger.warning(f"  ⏰ Timeout waiting for video frame at index {frames_yielded}")

        # PHASE 3: Continue generating remaining frames normally
        remaining_frames = num_frames_to_generate - frames_to_prefill
        if remaining_frames > 0:
            logger.debug(f"  🔄 Generating {remaining_frames} additional frames")
            for i in range(remaining_frames):
                silent_frame = self._create_silent_audio_frame()
                synced_audio_frames = self._audio_bstream.push(silent_frame.data)

                for synced_audio_frame in synced_audio_frames:
                    # Convert to float32 for Ditto
                    audio_data_int16 = np.frombuffer(synced_audio_frame.data, dtype=np.int16)
                    audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

                    # Add to Ditto's buffer
                    self._sdk_audio_buffer = np.concatenate([self._sdk_audio_buffer, audio_data_float])

                    # Process Ditto buffer when we have enough samples
                    if len(self._sdk_audio_buffer) >= self.split_len:
                        ditto_chunk = self._sdk_audio_buffer[: self.split_len]
                        self._sdk_audio_buffer = self._sdk_audio_buffer[self.chunksize[1] * 640 :]

                        logger.debug(f"  🎨 Feeding additional chunk to Ditto (queue: {self._video_frame_queue.qsize()})")
                        async with self._sdk_lock:
                            await self._loop.run_in_executor(
                                None, self.sdk.run_chunk, ditto_chunk, self.chunksize
                            )

                    # Yield this synced audio frame
                    yield synced_audio_frame
                    frames_yielded += 1

                    # Yield corresponding video frame
                    try:
                        video_frame = await asyncio.wait_for(
                            self._video_frame_queue.get(),
                            timeout=0.1
                        )
                        yield video_frame
                        frames_yielded += 1
                    except asyncio.TimeoutError:
                        logger.debug(f"  ⏰ Timeout at frame {frames_yielded}")

        logger.info(f"🏁 Idle generation complete: yielded {frames_yielded} total frames")

    def _create_silent_audio_frame(self) -> rtc.AudioFrame:
        """Create a silent audio frame for idle state."""
        silent_samples = self._options.audio_sample_rate // self._options.video_fps
        silent_data = np.zeros(silent_samples, dtype=np.int16)

        return rtc.AudioFrame(
            data=silent_data.tobytes(),
            sample_rate=self._options.audio_sample_rate,
            num_channels=self._options.audio_channels,
            samples_per_channel=silent_samples,
        )

    async def _flush_audio_buffer(self):
        """Flush remaining audio buffer to Ditto."""
        if len(self._sdk_audio_buffer) > 0:
            # Pad buffer to chunk size
            padding_len = self.split_len - (len(self._sdk_audio_buffer) % self.split_len)
            if padding_len < self.split_len:
                self._sdk_audio_buffer = np.concatenate([
                    self._sdk_audio_buffer,
                    np.zeros(padding_len, dtype=np.float32)
                ])

            # Process remaining chunks
            while len(self._sdk_audio_buffer) >= self.split_len:
                audio_chunk = self._sdk_audio_buffer[: self.split_len]
                self._sdk_audio_buffer = self._sdk_audio_buffer[self.split_len:]

                async with self._sdk_lock:
                    await self._loop.run_in_executor(
                        None, self.sdk.run_chunk, audio_chunk, self.chunksize
                    )

    async def aclose(self):
        """Cleanup resources."""
        logger.info("Closing DittoVideoGenerator...")
        async with self._sdk_lock:
            try:
                await self._loop.run_in_executor(
                    None, self.sdk.run_chunk, self.silent_chunk, self.chunksize
                )
                await self._loop.run_in_executor(None, self.sdk.close)
            except Exception as e:
                logger.error(f"Error closing Ditto SDK: {e}")
        logger.info("✅ DittoVideoGenerator closed")
