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
        # Initialize with prefix padding (3 frames = 1920 samples)
        self._sdk_audio_buffer = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
        self._pending_audio_chunks = []  # Store audio chunks to yield later
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
        # Increased buffer to 20 frames to accommodate pre-buffering and prevent draining
        self._video_frame_queue = asyncio.Queue[rtc.VideoFrame](maxsize=20)

        # Store event loop reference for thread-safe coroutine scheduling
        self._loop = asyncio.get_event_loop()

        # Diagnostic tracking for lip sync debugging
        self._frames_yielded = 0
        self._audio_frames_yielded = 0
        self._video_frames_yielded = 0
        self._generation_start_time = None

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

        # Log critical configuration values for debugging
        actual_valid_clip_len = self.sdk.audio2motion.valid_clip_len
        actual_overlap = self.sdk.audio2motion.overlap_v2
        actual_seq_frames = self.sdk.audio2motion.seq_frames
        logger.info(f"🔍 Ditto Config: seq_frames={actual_seq_frames}, overlap_v2={actual_overlap}, valid_clip_len={actual_valid_clip_len}")
        logger.info(f"🔍 Expected frames per chunk: {actual_valid_clip_len} (not chunksize[1]={self.chunksize[1]})")

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
        queue_before = self._video_frame_queue.qsize()
        logger.info(f"🎥 Callback: frame {frame_idx} at {timestamp:.3f}s (queue before: {queue_before})")
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
                logger.info(f"  ✅ Frame {frame_idx} queued (queue: {queue_before} → {self._video_frame_queue.qsize()})")
            except asyncio.QueueFull:
                logger.error(f"  ❌ Queue FULL! Dropping frame {frame_idx} (queue size: {self._video_frame_queue.qsize()}/{self._video_frame_queue.maxsize})")

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
        self._pending_audio_chunks.clear()

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
        """
        Frame-locked generation loop for perfect audio-video lip-sync with pre-buffering.

        Architecture:
        1. Pre-generate 15 idle frames to fill video queue (prevents cold start gaps)
        2. Accumulate audio to 6480 samples (first chunk: 405ms, subsequent: instant)
        3. Process through Ditto (~200ms, generates 5 NEW frames)
        4. Yield 5 audio-video pairs as fast as possible (no artificial pacing)
        5. Maintain 3280-sample overlap buffer for temporal coherence
        6. Video queue stays buffered at 10-15 frames to prevent draining

        Key Strategy:
        - Yield frames fast to keep queue full (LiveKit's synchronizer handles pacing)
        - Pre-buffered queue prevents starvation during Ditto processing gaps
        - Audio and video yielded together for frame-locked synchronization

        Latency: 605ms startup, ~400ms steady-state per audio sample
        Throughput: Sustained 25fps (queue-buffered)
        """

        # Pre-generate idle frames to warm up the video queue
        # This reduces "frame capture behind schedule" warnings during startup
        # With overlap_v2=70: each chunk produces 10 frames, so 2 chunks = 20 idle frames
        expected_prebuffer_frames = self.sdk.audio2motion.valid_clip_len * 2
        logger.info(f"Pre-generating idle frames to buffer video queue (expecting {expected_prebuffer_frames} frames)...")
        for i in range(2):  # Generate 2 chunks
            queue_before = self._video_frame_queue.qsize()
            silent_chunk = np.zeros(self.split_len, dtype=np.float32)
            async with self._sdk_lock:
                await self._loop.run_in_executor(
                    None, self.sdk.run_chunk, silent_chunk, self.chunksize
                )
            # Wait for callbacks to populate queue
            await asyncio.sleep(0.25)
            queue_after = self._video_frame_queue.qsize()
            logger.info(f"  Chunk {i+1}: {queue_after - queue_before} frames added (queue: {queue_before} → {queue_after})")
        actual_prebuffer = self._video_frame_queue.qsize()
        logger.info(f"✅ Pre-generated {actual_prebuffer} idle frames (expected {expected_prebuffer_frames})")
        if actual_prebuffer != expected_prebuffer_frames:
            logger.warning(f"⚠️ Pre-buffer mismatch: got {actual_prebuffer}, expected {expected_prebuffer_frames}")

        while True:
            # Step 1: Accumulate audio until we have enough for Ditto (6480 samples)
            # After first chunk, buffer has 3280 samples from overlap → only need 3200 more
            while len(self._sdk_audio_buffer) < self.split_len:
                try:
                    frame = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.04  # 40ms = 1 frame @ 25fps
                    )
                    self._audio_queue.task_done()
                    logger.debug(f"📥 Got frame: {type(frame).__name__}")

                    # Handle AudioSegmentEnd
                    if isinstance(frame, AudioSegmentEnd):
                        # Flush any pending audio chunks with their video
                        for audio_chunk in self._pending_audio_chunks:
                            yield audio_chunk
                            try:
                                video_frame = await asyncio.wait_for(
                                    self._video_frame_queue.get(), timeout=0.1
                                )
                                yield video_frame
                            except asyncio.TimeoutError:
                                logger.warning("Missing video frame during segment end")

                        # Signal segment end
                        yield AudioSegmentEnd()
                        self._pending_audio_chunks.clear()
                        continue

                    # Resample to 16kHz mono if necessary
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

                except asyncio.TimeoutError:
                    # No TTS audio - generate silent frame for idle animation
                    logger.debug("⚪ Generating silent frame (idle)")
                    silent_frame = self._create_silent_audio_frame()
                    resampled_frames = [silent_frame]

                # Chunk audio into 640-sample frames and buffer for Ditto
                for resampled_frame in resampled_frames:
                    synced_audio_frames = self._audio_bstream.push(resampled_frame.data)

                    for synced_audio_frame in synced_audio_frames:
                        # Convert to float32 for Ditto
                        audio_data_int16 = np.frombuffer(synced_audio_frame.data, dtype=np.int16)
                        audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

                        # Add to Ditto buffer
                        self._sdk_audio_buffer = np.concatenate([
                            self._sdk_audio_buffer,
                            audio_data_float
                        ])

                        # Store for yielding later (after video is ready)
                        self._pending_audio_chunks.append(synced_audio_frame)

            # Step 2: Process through Ditto
            ditto_chunk = self._sdk_audio_buffer[:self.split_len]
            # Use dynamic valid_clip_len which accounts for overlap_v2 configuration
            # With overlap_v2=70: valid_clip_len = seq_frames(80) - overlap_v2(70) = 10 frames
            # With overlap_v2=10: valid_clip_len = seq_frames(80) - overlap_v2(10) = 70 frames
            expected_video_frames = self.sdk.audio2motion.valid_clip_len

            queue_before = self._video_frame_queue.qsize()
            logger.debug(f"🎨 Feeding {len(ditto_chunk)} samples to Ditto (expect {expected_video_frames} frames, queue before: {queue_before})")
            await self._process_ditto_chunk(ditto_chunk, expected_video_frames)
            queue_after = self._video_frame_queue.qsize()
            frames_added = queue_after - queue_before

            logger.info(f"📦 Ditto produced {frames_added} frames (expected {expected_video_frames}), queue: {queue_before} → {queue_after}")
            if frames_added != expected_video_frames:
                logger.error(f"❌ MISMATCH: Ditto produced {frames_added} frames but expected {expected_video_frames}!")

            # Step 3: Yield audio-video pairs as fast as possible
            # Let LiveKit's AvatarRunner synchronizer handle the pacing (it has its own 25fps timer)
            # Our job: keep the queue full so synchronizer never starves
            if self._generation_start_time is None:
                self._generation_start_time = time.time()

            for i in range(expected_video_frames):
                if i >= len(self._pending_audio_chunks):
                    logger.warning(f"⚠️ Ran out of audio chunks at {i}/{expected_video_frames}")
                    break

                # Yield audio
                yield self._pending_audio_chunks[i]
                self._audio_frames_yielded += 1

                # Yield corresponding video (synchronized)
                try:
                    video_frame = await asyncio.wait_for(
                        self._video_frame_queue.get(),
                        timeout=0.1
                    )
                    yield video_frame
                    self._video_frames_yielded += 1

                    # Log every 25 frames (1 second worth) for diagnostics
                    if self._video_frames_yielded % 25 == 0:
                        elapsed = time.time() - self._generation_start_time
                        expected_time = self._video_frames_yielded / 25.0  # Expected at 25fps
                        drift = elapsed - expected_time
                        logger.info(
                            f"📊 Sync check: {self._video_frames_yielded} frames in {elapsed:.2f}s "
                            f"(expected {expected_time:.2f}s, drift: {drift:+.2f}s, queue: {self._video_frame_queue.qsize()})"
                        )
                except asyncio.TimeoutError:
                    logger.error(f"❌ Missing video frame {i+1}/{expected_video_frames}")
                    break

            # Step 4: Advance buffers (maintain overlap for temporal coherence)
            # CRITICAL: Advance by chunksize[1] (step size), NOT valid_clip_len (output frames)!
            # With overlap_v2=70: chunksize[1]=5 (step), valid_clip_len=10 (output)
            # We must step by 5 frames (3200 samples) to maintain proper overlap for Ditto
            step_samples = self.chunksize[1] * 640  # 5 * 640 = 3200 samples
            self._sdk_audio_buffer = self._sdk_audio_buffer[step_samples:]

            # Remove yielded audio chunks, keep any excess for next cycle
            self._pending_audio_chunks = self._pending_audio_chunks[expected_video_frames:]

    async def _process_ditto_chunk(self, ditto_chunk: np.ndarray, expected_frames: int):
        """Process an audio chunk through Ditto and wait for video frames."""
        start_time = time.time()
        queue_before = self._video_frame_queue.qsize()

        async with self._sdk_lock:
            await self._loop.run_in_executor(
                None, self.sdk.run_chunk, ditto_chunk, self.chunksize
            )
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"✅ Ditto complete in {elapsed:.1f}ms (queue: {self._video_frame_queue.qsize()})")

        # Wait for NEW frames to be added (not total queue size)
        # Callbacks can be slightly delayed by thread scheduling, so be generous with timeout
        target_queue_size = queue_before + expected_frames
        timeout_deadline = time.time() + 0.3  # 300ms max wait
        while self._video_frame_queue.qsize() < target_queue_size:
            if time.time() > timeout_deadline:
                current_size = self._video_frame_queue.qsize()
                frames_added = current_size - queue_before
                logger.warning(f"⏰ Only {frames_added}/{expected_frames} new frames added after 300ms (queue: {queue_before} → {current_size})")
                break
            await asyncio.sleep(0.01)  # 10ms polling interval

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
