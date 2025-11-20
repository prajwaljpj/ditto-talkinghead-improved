import sys
import os
import asyncio
import logging
import numpy as np
import cv2
import time
from typing import Optional, Union
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

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
        frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    yuv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv_frame.tobytes()


@dataclass
class TimestampedPair:
    """Audio-video pair with timestamp."""
    timestamp_ms: int
    audio: Optional[rtc.AudioFrame] = None
    video: Optional[rtc.VideoFrame] = None

    def is_complete(self) -> bool:
        return self.audio is not None and self.video is not None

    def __repr__(self) -> str:
        a = "✓" if self.audio else "✗"
        v = "✓" if self.video else "✗"
        return f"Pair({self.timestamp_ms}ms: a={a}, v={v})"


class DittoVideoGeneratorDecoupled(VideoGenerator):
    """
    Decoupled architecture: Processing and yielding are INDEPENDENT.

    - Task 1 (background): Accumulate audio continuously
    - Task 2 (background): Process chunks continuously (don't wait for frames!)
    - Task 3 (main): Yield frames as they become available

    No more waiting! Frames arrive async, we yield them when ready.
    """

    def __init__(self, options: AvatarOptions, data_root: str, cfg_pkl: str, source_path: str):
        self._options = options
        self._audio_queue = asyncio.Queue[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._audio_resampler: Optional[rtc.AudioResampler] = None

        # Ditto config
        self.chunksize = (3, 5, 2)
        self.split_len = 6480
        self.ditto_sample_rate = 16000
        self.silent_chunk = np.zeros(self.split_len, dtype=np.float32)

        # Audio buffering
        self._sdk_audio_buffer = np.zeros((0,), dtype=np.float32)
        self._sdk_lock = asyncio.Lock()

        # AudioByteStream
        samples_per_frame = options.audio_sample_rate // options.video_fps
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=samples_per_frame,
        )

        # Timestamp matching
        self._pending_pairs = {}
        self._current_audio_timestamp_ms = 0
        self._next_timestamp_to_yield_ms = 0
        self._first_chunk = True

        # Flow control to prevent latency buildup and ensure smooth streaming
        self.MAX_PAIRS_PER_BATCH = 5  # Small batches prevent flooding LiveKit buffer
        self.MAX_PENDING_PAIRS = 25   # ~1 second max latency (25 fps)

        # Events
        self._loop = asyncio.get_event_loop()
        self._pair_ready_event = asyncio.Event()  # Signals complete pair available
        self._buffer_ready_event = asyncio.Event()  # Signals buffer >= split_len
        self._stop_event = asyncio.Event()

        # Background tasks
        self._audio_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None

        # Diagnostics
        self._chunks_processed = 0
        self._frames_yielded = 0
        self._generation_start = None

        # Wait time diagnostics
        self._audio_queue_wait_times = []
        self._audio_lock_wait_times = []
        self._buffer_event_wait_times = []
        self._processing_lock1_wait_times = []
        self._ditto_processing_times = []
        self._processing_lock2_wait_times = []
        self._pair_event_wait_times = []

        # Initialize Ditto
        logger.info("Initializing Ditto StreamSDK...")
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.sdk.setup(
            source_path,
            output_path="/dev/null",
            frame_callback=self._handle_frame,
            online_mode=True,
            fps=options.video_fps,
        )
        self.sdk.setup_Nd(N_d=1000000)

        logger.info(f"Ditto: seq_frames={self.sdk.audio2motion.seq_frames}, "
                   f"overlap={self.sdk.audio2motion.overlap_v2}, "
                   f"valid_len={self.sdk.audio2motion.valid_clip_len}")

        # Warmup
        logger.info("Warming up...")
        for i in range(3):
            self.sdk.run_chunk(self.silent_chunk, self.chunksize)
        self._pending_pairs.clear()
        logger.info("✅ Ready")

    def _handle_frame(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """Video frame callback from Ditto."""
        try:
            timestamp_ms = round(timestamp * 1000)

            i420 = rgb_to_i420(frame_rgb, self._options.video_width, self._options.video_height)
            video = rtc.VideoFrame(
                data=i420,
                width=self._options.video_width,
                height=self._options.video_height,
                type=rtc.VideoBufferType.I420,
            )

            # Store video
            if timestamp_ms not in self._pending_pairs:
                self._pending_pairs[timestamp_ms] = TimestampedPair(timestamp_ms)
            self._pending_pairs[timestamp_ms].video = video

            # Signal if pair is complete
            pair = self._pending_pairs[timestamp_ms]
            if pair.is_complete():
                self._loop.call_soon_threadsafe(self._pair_ready_event.set)
                logger.debug(f"✅ Complete pair at {timestamp_ms}ms")

        except Exception as e:
            logger.error(f"Frame callback error: {e}", exc_info=True)

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        logger.info("Clearing buffer")
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._audio_bstream.flush()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._main_loop()

    async def _audio_accumulation_task(self):
        """Background: Continuously accumulate audio."""
        logger.info("🎧 Audio task started")
        try:
            while not self._stop_event.is_set():
                try:
                    # Measure queue wait time
                    t_queue_start = time.time()
                    frame = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.04
                    )
                    queue_wait_ms = (time.time() - t_queue_start) * 1000
                    self._audio_queue_wait_times.append(queue_wait_ms)

                    if isinstance(frame, AudioSegmentEnd):
                        logger.info("AudioSegmentEnd in audio task")
                        continue

                    # Resample
                    resampled = []
                    if frame.sample_rate != self.ditto_sample_rate or frame.num_channels != 1:
                        if not self._audio_resampler:
                            self._audio_resampler = rtc.AudioResampler(
                                input_rate=frame.sample_rate,
                                output_rate=self.ditto_sample_rate,
                                num_channels=1,
                            )
                        for f in self._audio_resampler.push(frame):
                            resampled.append(f)
                    else:
                        resampled.append(frame)

                except asyncio.TimeoutError:
                    # Silent frame - queue wait was full timeout
                    self._audio_queue_wait_times.append(40.0)  # Full timeout
                    silent = self._create_silent_frame()
                    resampled = [silent]

                # Chunk and buffer - measure lock wait time
                t_lock_start = time.time()
                async with self._sdk_lock:
                    lock_wait_ms = (time.time() - t_lock_start) * 1000
                    self._audio_lock_wait_times.append(lock_wait_ms)
                    for rf in resampled:
                        for synced in self._audio_bstream.push(rf.data):
                            # Convert to float32
                            data_int16 = np.frombuffer(synced.data, dtype=np.int16)
                            data_float = data_int16.astype(np.float32) / 32768.0

                            # Add to buffer
                            self._sdk_audio_buffer = np.concatenate([
                                self._sdk_audio_buffer, data_float
                            ])

                            # Create audio pair
                            ts = self._current_audio_timestamp_ms
                            if ts not in self._pending_pairs:
                                self._pending_pairs[ts] = TimestampedPair(ts)
                            self._pending_pairs[ts].audio = synced

                            # Signal if complete
                            if self._pending_pairs[ts].is_complete():
                                self._pair_ready_event.set()

                            self._current_audio_timestamp_ms += 40

                    # Signal if buffer ready
                    if len(self._sdk_audio_buffer) >= self.split_len:
                        self._buffer_ready_event.set()

        except Exception as e:
            logger.error(f"Audio task error: {e}", exc_info=True)
        finally:
            logger.info("🛑 Audio task stopped")

    async def _chunk_processing_task(self):
        """Background: Process chunks continuously (DON'T wait for frames!)."""
        logger.info("🎨 Processing task started")
        try:
            while not self._stop_event.is_set():
                # Measure buffer event wait time
                t_event_start = time.time()
                await self._buffer_ready_event.wait()
                event_wait_ms = (time.time() - t_event_start) * 1000
                self._buffer_event_wait_times.append(event_wait_ms)

                # Measure first lock wait time
                t_lock1_start = time.time()
                async with self._sdk_lock:
                    lock1_wait_ms = (time.time() - t_lock1_start) * 1000
                    self._processing_lock1_wait_times.append(lock1_wait_ms)

                    if len(self._sdk_audio_buffer) < self.split_len:
                        self._buffer_ready_event.clear()
                        continue

                    # Add padding for first chunk
                    if self._first_chunk:
                        padding = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
                        self._sdk_audio_buffer = np.concatenate([padding, self._sdk_audio_buffer])
                        logger.info("🎬 First chunk: added padding")
                        self._first_chunk = False

                    # Extract chunk
                    chunk = self._sdk_audio_buffer[:self.split_len].copy()

                # Process (releases lock!) - measure Ditto processing time
                t_ditto_start = time.time()
                await self._loop.run_in_executor(
                    None, self.sdk.run_chunk, chunk, self.chunksize
                )
                ditto_time_ms = (time.time() - t_ditto_start) * 1000
                self._ditto_processing_times.append(ditto_time_ms)
                self._chunks_processed += 1
                logger.info(f"✅ Chunk {self._chunks_processed} processed in {ditto_time_ms:.1f}ms")

                # Measure second lock wait time
                t_lock2_start = time.time()
                async with self._sdk_lock:
                    lock2_wait_ms = (time.time() - t_lock2_start) * 1000
                    self._processing_lock2_wait_times.append(lock2_wait_ms)
                    step = self.chunksize[1] * 640
                    self._sdk_audio_buffer = self._sdk_audio_buffer[step:]

                    if len(self._sdk_audio_buffer) < self.split_len:
                        self._buffer_ready_event.clear()

                # Small delay to yield control
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Processing task error: {e}", exc_info=True)
        finally:
            logger.info("🛑 Processing task stopped")

    async def _main_loop(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        """
        Main loop: Yield frames as they become available.

        NO WAITING! Just yield whatever is ready.
        """
        logger.info("Starting main loop")

        # Start background tasks
        self._audio_task = asyncio.create_task(self._audio_accumulation_task())
        self._processing_task = asyncio.create_task(self._chunk_processing_task())

        try:
            self._generation_start = time.time()
            batch_count = 0

            while True:
                # Wait for a complete pair to be available
                batch_total_start = time.time()
                wait_start = time.time()
                await self._pair_ready_event.wait()
                wait_duration = (time.time() - wait_start) * 1000
                self._pair_event_wait_times.append(wait_duration)

                batch_count += 1
                pairs_in_batch = 0
                yield_total_time = 0

                # Yield all consecutive complete pairs starting from cursor
                inner_loop_iterations = 0
                while True:
                    inner_loop_iterations += 1
                    ts = self._next_timestamp_to_yield_ms
                    pair = self._pending_pairs.get(ts)

                    # Break if: (1) no pair, (2) incomplete pair, OR (3) batch limit reached
                    if pair is None or not pair.is_complete() or pairs_in_batch >= self.MAX_PAIRS_PER_BATCH:
                        # Clear event only if no more complete pairs
                        if pair is None or not pair.is_complete():
                            self._pair_ready_event.clear()

                        # Log batch statistics (ALWAYS, to see the pattern)
                        batch_total_time = (time.time() - batch_total_start) * 1000
                        reason = "incomplete" if (pair is None or not pair.is_complete()) else "batch_limit"
                        logger.info(
                            f"🔄 Batch {batch_count}: total={batch_total_time:.1f}ms "
                            f"(wait={wait_duration:.1f}ms, yield={yield_total_time:.1f}ms), "
                            f"pairs={pairs_in_batch}, inner_iters={inner_loop_iterations}, "
                            f"next_ts={self._next_timestamp_to_yield_ms}ms, reason={reason}"
                        )
                        break

                    # Safety: Log if inner loop runs too long
                    if inner_loop_iterations % 100 == 0:
                        logger.warning(
                            f"⚠️ Inner loop running continuously: {inner_loop_iterations} iterations, "
                            f"pairs_yielded={pairs_in_batch}"
                        )

                    # Yield pair (audio first, then video - maintains lip sync!)
                    t_yield_start = time.time()
                    yield pair.audio
                    yield pair.video

                    # CRITICAL: Yield control to event loop so LiveKit can send frames over network
                    # Without this, we flood the buffer and cause stuttering
                    await asyncio.sleep(0)

                    yield_duration = (time.time() - t_yield_start) * 1000
                    yield_total_time += yield_duration

                    # Cleanup
                    del self._pending_pairs[ts]
                    self._next_timestamp_to_yield_ms += 40
                    self._frames_yielded += 1
                    pairs_in_batch += 1

                    # Log slow yields
                    if yield_duration > 50:  # Slower than expected (40ms)
                        logger.warning(f"⏱️ Slow yield: {yield_duration:.1f}ms at {ts}ms")

                    # Diagnostics every second
                    if self._frames_yielded % 25 == 0:
                        elapsed = time.time() - self._generation_start
                        expected = self._frames_yielded / 25.0
                        drift = elapsed - expected

                        # Count complete vs incomplete pairs
                        complete = sum(1 for p in self._pending_pairs.values() if p.is_complete())
                        incomplete = len(self._pending_pairs) - complete

                        # Calculate latency (pending × 40ms)
                        latency_ms = len(self._pending_pairs) * 40

                        logger.info(
                            f"📊 {self._frames_yielded} frames in {elapsed:.2f}s "
                            f"(drift: {drift:+.2f}s, pending: {len(self._pending_pairs)} "
                            f"[{complete} complete, {incomplete} incomplete], "
                            f"latency: {latency_ms}ms)"
                        )

                        # Warn if latency too high
                        if len(self._pending_pairs) > self.MAX_PENDING_PAIRS:
                            logger.warning(
                                f"⚠️ High latency! {len(self._pending_pairs)} pending pairs "
                                f"({latency_ms}ms buffered) - target is {self.MAX_PENDING_PAIRS} pairs "
                                f"({self.MAX_PENDING_PAIRS * 40}ms)"
                            )

                        # Wait time statistics
                        self._log_wait_statistics()

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            raise
        finally:
            # Stop background tasks
            self._stop_event.set()
            if self._audio_task:
                await self._audio_task
            if self._processing_task:
                await self._processing_task

    def _log_wait_statistics(self):
        """Log wait time statistics for all waits."""

        def stats(times, name):
            if not times:
                return f"{name}: no data"
            avg = sum(times) / len(times)
            max_t = max(times)
            min_t = min(times)
            # Count very short waits (< 0.1ms = didn't actually wait)
            instant = sum(1 for t in times if t < 0.1)
            return f"{name}: avg={avg:.2f}ms, max={max_t:.2f}ms, min={min_t:.2f}ms, instant={instant}/{len(times)}"

        logger.info("⏱️  WAIT TIME STATISTICS:")
        logger.info(f"   Audio Task:")
        logger.info(f"     - {stats(self._audio_queue_wait_times, 'Queue wait')}")
        logger.info(f"     - {stats(self._audio_lock_wait_times, 'Lock wait')}")
        logger.info(f"   Processing Task:")
        logger.info(f"     - {stats(self._buffer_event_wait_times, 'Buffer event wait')}")
        logger.info(f"     - {stats(self._processing_lock1_wait_times, 'Lock1 wait')}")
        logger.info(f"     - {stats(self._ditto_processing_times, 'Ditto processing')}")
        logger.info(f"     - {stats(self._processing_lock2_wait_times, 'Lock2 wait')}")
        logger.info(f"   Main Loop:")
        logger.info(f"     - {stats(self._pair_event_wait_times, 'Pair event wait')} ⚠️ KEY METRIC")

        # Clear arrays to avoid memory growth
        self._audio_queue_wait_times.clear()
        self._audio_lock_wait_times.clear()
        self._buffer_event_wait_times.clear()
        self._processing_lock1_wait_times.clear()
        self._ditto_processing_times.clear()
        self._processing_lock2_wait_times.clear()
        self._pair_event_wait_times.clear()

    def _create_silent_frame(self) -> rtc.AudioFrame:
        """Create silent audio frame."""
        samples = self._options.audio_sample_rate // self._options.video_fps
        data = np.zeros(samples, dtype=np.int16)
        return rtc.AudioFrame(
            data=data.tobytes(),
            sample_rate=self._options.audio_sample_rate,
            num_channels=self._options.audio_channels,
            samples_per_channel=samples,
        )

    async def aclose(self):
        """Cleanup."""
        logger.info("Closing...")
        self._stop_event.set()

        if self._audio_task:
            await self._audio_task
        if self._processing_task:
            await self._processing_task

        async with self._sdk_lock:
            try:
                await self._loop.run_in_executor(None, self.sdk.close)
            except Exception as e:
                logger.error(f"Close error: {e}")

        logger.info("✅ Closed")
