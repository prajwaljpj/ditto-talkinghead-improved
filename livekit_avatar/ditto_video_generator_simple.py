"""
Simplified Ditto Video Generator - Following LiveKit Example Pattern

Key insight from LiveKit's audio_wave example:
1. No complex state machine
2. Video generated synchronously from audio
3. Yield video before audio
4. Let AVSynchronizer handle timing

For Ditto, we adapt this by:
1. Buffer audio until we can process a chunk
2. Process chunk → get video frames
3. Pair and yield (video first, then audio)
4. No rate limiting - AVSynchronizer handles it
"""

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

logger = logging.getLogger(__name__)
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


class DittoVideoGeneratorSimple(VideoGenerator):
    """
    Simplified Ditto generator following LiveKit's pattern.
    
    Flow:
    1. Receive audio from TTS
    2. Buffer until chunk ready
    3. Process through Ditto (blocking)
    4. Yield video+audio pairs
    5. On silence, generate idle frames
    """

    def __init__(self, options: AvatarOptions, data_root: str, cfg_pkl: str, source_path: str):
        self._options = options
        self._audio_queue = asyncio.Queue[Union[rtc.AudioFrame, AudioSegmentEnd]]()
        self._audio_resampler: Optional[rtc.AudioResampler] = None

        # Ditto config
        self.chunksize = (3, 5, 2)  # (padding, output, overlap) frames
        self.split_len = 6480  # Samples needed for one chunk
        self.ditto_sample_rate = 16000

        # Audio byte stream for frame alignment
        samples_per_frame = options.audio_sample_rate // options.video_fps
        self._audio_bstream = utils.audio.AudioByteStream(
            sample_rate=options.audio_sample_rate,
            num_channels=options.audio_channels,
            samples_per_channel=samples_per_frame,
        )

        # Buffers
        self._audio_buffer = np.zeros((0,), dtype=np.float32)  # For Ditto
        self._audio_frames = []  # Aligned audio frames for output
        self._video_frames = []  # Video frames from Ditto
        
        # Timing
        self._frame_duration_ms = 1000.0 / options.video_fps
        self._last_idle_time = 0.0
        
        # First chunk needs extra padding
        self._first_chunk = True
        
        # Diagnostics
        self._frames_yielded = 0
        self._start_time = None

        # Initialize Ditto
        logger.info("Initializing Ditto StreamSDK...")
        self._loop = asyncio.get_event_loop()
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.sdk.setup(
            source_path,
            output_path="/dev/null",
            frame_callback=self._on_video_frame,
            online_mode=True,
            fps=options.video_fps,
        )
        self.sdk.setup_Nd(N_d=1000000)

        # Warmup Ditto (sets d0 reference)
        logger.info("Warming up Ditto...")
        silent_chunk = np.zeros(self.split_len, dtype=np.float32)
        for _ in range(3):
            self.sdk.run_chunk(silent_chunk, self.chunksize)
        self._video_frames.clear()  # Discard warmup frames
        logger.info("✅ Ready")

    def _on_video_frame(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """Callback from Ditto - collect video frames."""
        i420 = rgb_to_i420(frame_rgb, self._options.video_width, self._options.video_height)
        video = rtc.VideoFrame(
            data=i420,
            width=self._options.video_width,
            height=self._options.video_height,
            type=rtc.VideoBufferType.I420,
        )
        self._video_frames.append(video)

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._audio_bstream.flush()
        self._audio_buffer = np.zeros((0,), dtype=np.float32)
        self._audio_frames.clear()
        self._video_frames.clear()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._generate()

    async def _generate(self) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        """
        Main generation loop - simple and direct.
        
        Pattern from LiveKit example:
        1. Get audio
        2. Generate video
        3. Yield video, then audio
        """
        logger.info("Starting generation loop")
        self._start_time = time.time()

        while True:
            try:
                # Wait for audio with short timeout
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), 
                    timeout=self._frame_duration_ms / 1000.0  # ~40ms
                )
                
                if isinstance(frame, AudioSegmentEnd):
                    # Flush any remaining audio
                    async for pair in self._flush_and_yield():
                        yield pair
                    yield AudioSegmentEnd()
                    continue
                
                # Process real audio
                async for pair in self._process_audio_frame(frame):
                    yield pair
                    
            except asyncio.TimeoutError:
                # No audio - generate idle frame
                async for pair in self._generate_idle_frame():
                    yield pair

    async def _process_audio_frame(self, frame: rtc.AudioFrame) -> AsyncGenerator:
        """Process an audio frame through Ditto pipeline."""
        # Resample if needed
        resampled = self._resample(frame)
        
        for rf in resampled:
            # Get aligned audio frames
            for synced in self._audio_bstream.push(rf.data):
                # Convert for Ditto
                data_int16 = np.frombuffer(synced.data, dtype=np.int16)
                data_float = data_int16.astype(np.float32) / 32768.0
                self._audio_buffer = np.concatenate([self._audio_buffer, data_float])
                self._audio_frames.append(synced)
        
        # Process if we have enough audio
        while len(self._audio_buffer) >= self.split_len:
            # Add padding for first chunk - ONLY to Ditto buffer, not audio frames
            # Padding is context for Ditto, it doesn't produce video frames
            if self._first_chunk:
                padding = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
                self._audio_buffer = np.concatenate([padding, self._audio_buffer])
                self._first_chunk = False
                logger.info("🎬 First chunk: added context padding to Ditto buffer only")
            
            # Extract and process chunk
            chunk = self._audio_buffer[:self.split_len].copy()
            
            # Process through Ditto (blocking)
            await self._loop.run_in_executor(None, self.sdk.run_chunk, chunk, self.chunksize)
            
            # Advance buffer
            step = self.chunksize[1] * 640  # 5 frames * 640 samples
            self._audio_buffer = self._audio_buffer[step:]
            
            # Yield paired frames (video first, then audio!)
            num_pairs = min(len(self._video_frames), len(self._audio_frames))
            for _ in range(num_pairs):
                video = self._video_frames.pop(0)
                audio = self._audio_frames.pop(0)
                yield video
                yield audio
                self._frames_yielded += 1
            
            # Log progress
            if self._frames_yielded % 25 == 0:
                elapsed = time.time() - self._start_time
                fps = self._frames_yielded / elapsed if elapsed > 0 else 0
                logger.info(f"📊 {self._frames_yielded} frames | {fps:.1f}fps")

    async def _generate_idle_frame(self) -> AsyncGenerator:
        """Generate a single idle frame (rate-limited)."""
        now = time.time()
        if (now - self._last_idle_time) * 1000 < self._frame_duration_ms * 0.9:
            return  # Too soon
        self._last_idle_time = now
        
        # Add silent audio to buffer
        silent_samples = np.zeros(640, dtype=np.float32)
        self._audio_buffer = np.concatenate([self._audio_buffer, silent_samples])
        self._audio_frames.append(self._create_silent_frame())
        
        # Process if buffer ready
        if len(self._audio_buffer) >= self.split_len:
            if self._first_chunk:
                padding = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
                self._audio_buffer = np.concatenate([padding, self._audio_buffer])
                self._first_chunk = False
            
            chunk = self._audio_buffer[:self.split_len].copy()
            await self._loop.run_in_executor(None, self.sdk.run_chunk, chunk, self.chunksize)
            
            step = self.chunksize[1] * 640
            self._audio_buffer = self._audio_buffer[step:]
            
            # Yield pairs
            num_pairs = min(len(self._video_frames), len(self._audio_frames))
            for _ in range(num_pairs):
                video = self._video_frames.pop(0)
                audio = self._audio_frames.pop(0)
                yield video
                yield audio
                self._frames_yielded += 1

    async def _flush_and_yield(self) -> AsyncGenerator:
        """Flush remaining buffer at end of speech."""
        if len(self._audio_buffer) > 0:
            # Pad to complete chunk - this is TRAILING padding, not context
            # We DO need silent audio frames for trailing padding since Ditto will produce video
            if len(self._audio_buffer) < self.split_len:
                padding_needed = self.split_len - len(self._audio_buffer)
                padding = np.zeros(padding_needed, dtype=np.float32)
                self._audio_buffer = np.concatenate([self._audio_buffer, padding])
                # Add silent frames for the padding portion that will produce video
                # Ditto outputs 5 frames per chunk, padding at end produces neutral video
                padding_frames = (padding_needed + 639) // 640
                for _ in range(padding_frames):
                    self._audio_frames.append(self._create_silent_frame())
            
            # First chunk context padding - doesn't produce video, don't add audio frames
            if self._first_chunk:
                padding = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
                self._audio_buffer = np.concatenate([padding, self._audio_buffer])
                self._first_chunk = False
            
            chunk = self._audio_buffer[:self.split_len].copy()
            await self._loop.run_in_executor(None, self.sdk.run_chunk, chunk, self.chunksize)
            self._audio_buffer = np.zeros((0,), dtype=np.float32)
            
            # Yield all remaining pairs
            num_pairs = min(len(self._video_frames), len(self._audio_frames))
            for _ in range(num_pairs):
                video = self._video_frames.pop(0)
                audio = self._audio_frames.pop(0)
                yield video
                yield audio
                self._frames_yielded += 1
        
        # Clear any remaining (context offset - expected)
        if self._audio_frames:
            logger.debug(f"Clearing {len(self._audio_frames)} context offset audio frames")
        if self._video_frames:
            logger.warning(f"Unexpected: {len(self._video_frames)} orphaned video frames")
        self._video_frames.clear()
        self._audio_frames.clear()

    def _resample(self, frame: rtc.AudioFrame) -> list:
        """Resample audio if needed."""
        if frame.sample_rate != self.ditto_sample_rate or frame.num_channels != 1:
            if not self._audio_resampler:
                self._audio_resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=self.ditto_sample_rate,
                    num_channels=1,
                )
            return list(self._audio_resampler.push(frame))
        return [frame]

    def _create_silent_frame(self) -> rtc.AudioFrame:
        """Create a silent audio frame."""
        samples = self._options.audio_sample_rate // self._options.video_fps
        return rtc.AudioFrame(
            data=np.zeros(samples, dtype=np.int16).tobytes(),
            sample_rate=self._options.audio_sample_rate,
            num_channels=self._options.audio_channels,
            samples_per_channel=samples,
        )

    async def aclose(self):
        """Cleanup."""
        logger.info("Closing...")
        try:
            await self._loop.run_in_executor(None, self.sdk.close)
        except Exception as e:
            logger.error(f"Close error: {e}")
        logger.info("✅ Closed")

