import sys
import os
import asyncio
import logging
import numpy as np
import cv2
import time
from typing import Optional
from livekit import rtc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust system path to import stream_pipeline_online from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from stream_pipeline_online import StreamSDK
except ImportError:
    raise ImportError(
        "StreamSDK not found. Ensure 'stream_pipeline_online.py' is in the project root folder."
    )

# --- Configuration ---
FPS = 50


def rgb_to_i420(frame_rgb: np.ndarray, width: int, height: int) -> bytes:
    """Converts an RGB NumPy array to I420 byte data, resizing if necessary."""
    if frame_rgb.shape[0] != height or frame_rgb.shape[1] != width:
        frame_rgb = cv2.resize(
            frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR
        )

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    yuv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    return yuv_frame.tobytes()


class CustomAvatarWorker:
    """
    Worker that generates avatar video frames driven by audio input.

    This worker:
    1. Receives audio frames (TTS output from the agent)
    2. Feeds them to the Ditto avatar generation model (StreamSDK)
    3. Receives generated video frames and publishes them to AVSynchronizer
       with timestamps for perfect audio/video synchronization

    The worker operates in different states:
    - idle: Generates subtle idle animations (breathing, blinking)
    - listening: User is speaking (slight head movements)
    - thinking: Agent is processing (contemplative pose)
    - speaking: Agent is speaking (lip-sync with TTS audio)

    Synchronization:
    - Video frames are timestamped based on frame_idx from Ditto
    - Timestamps match the corresponding audio that generated them
    - AVSynchronizer ensures both tracks play in perfect sync
    """

    def __init__(
        self,
        data_root: str,
        cfg_pkl: str,
        source_path: str,
        frame_width: int,
        frame_height: int,
        av_sync: rtc.AVSynchronizer,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.av_sync = av_sync

        # Audio processing configuration
        self.chunksize = (3, 5, 2)  # Ditto-specific chunking
        self.split_len = 6480  # Samples per chunk for 16kHz audio
        self.sample_rate = 16000
        self.silent_chunk = np.zeros(self.split_len, dtype=np.float32)

        # State management
        self.current_state = "idle"
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None

        # Audio buffering and timing
        self._audio_queue = asyncio.Queue(maxsize=100)
        self._sdk_lock = asyncio.Lock()
        self._base_timestamp = None  # Track base timestamp for sync
        self._frame_duration = 1.0 / FPS  # Duration of each frame
        self._audio_samples_processed = (
            0  # Track total audio samples for timestamp calculation
        )

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
            fps=FPS,
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
        Converts and publishes video frames to AVSynchronizer with timestamps.
        """
        try:
            i420_data = rgb_to_i420(frame_rgb, self.frame_width, self.frame_height)

            video_frame = rtc.VideoFrame(
                data=i420_data,
                width=self.frame_width,
                height=self.frame_height,
                type=rtc.VideoBufferType.I420,
            )

            # Calculate synchronized timestamp
            # Use frame_idx to compute timestamp relative to base
            if self._base_timestamp is None:
                self._base_timestamp = time.time()

            # Timestamp based on frame index for consistent timing
            sync_timestamp = self._base_timestamp + (frame_idx * self._frame_duration)

            # Push frame to AVSynchronizer with timestamp
            # Note: We need to schedule this in the event loop since we're in a thread
            # Use the stored event loop reference (not get_event_loop() which fails in threads)
            future = asyncio.run_coroutine_threadsafe(
                self.av_sync.push(video_frame, sync_timestamp), self._loop
            )
            # Optionally wait for completion (with timeout to avoid blocking)
            try:
                future.result(timeout=0.2)  # Increased timeout
            except Exception as e:
                logger.debug(f"Frame push timeout (frame {frame_idx}): {e}")

        except Exception as e:
            logger.error(f"Error handling generated frame: {e}")

    def _warmup_model(self):
        """
        Warm up the Ditto model by generating a few dummy frames.
        This initializes CUDA, loads TensorRT engines, etc.
        """
        # Temporarily disable frame callback during warmup
        original_callback = (
            self.sdk._handle_generated_frame
            if hasattr(self.sdk, "_handle_generated_frame")
            else None
        )

        # Generate 3-5 warmup frames with silent audio
        warmup_audio = np.zeros(self.split_len, dtype=np.float32)
        for i in range(3):
            try:
                self.sdk.run_chunk(warmup_audio, self.chunksize)
            except Exception as e:
                logger.warning(f"Warmup frame {i} failed: {e}")

        # Reset timing tracking after warmup
        self._base_timestamp = None
        self._audio_samples_processed = 0

    def start(self):
        """Starts the avatar worker's processing loops."""
        if self._running:
            logger.warning("Avatar worker already running")
            return

        self._running = True

        # Start background tasks
        loop = asyncio.get_event_loop()
        self._processing_task = loop.create_task(self._run_audio_processing())
        logger.info("✅ Avatar worker processing started")

    async def feed_audio(self, audio_frame: rtc.AudioFrame):
        """
        Feed a TTS audio frame to the avatar for lip-sync.

        Args:
            audio_frame: Audio frame from the agent's TTS output
        """
        try:
            # Convert audio frame to float32 numpy array
            audio_data_int16 = np.frombuffer(audio_frame.data, dtype=np.int16)
            audio_data_float = audio_data_int16.astype(np.float32) / 32768.0

            # Resample if necessary (Ditto expects 16kHz)
            if audio_frame.sample_rate != self.sample_rate:
                try:
                    import scipy.signal

                    ratio = self.sample_rate / audio_frame.sample_rate
                    num_samples = int(len(audio_data_float) * ratio)
                    audio_data_float = scipy.signal.resample(
                        audio_data_float, num_samples
                    )
                except ImportError:
                    logger.warning(
                        f"scipy not available for resampling. "
                        f"Expected {self.sample_rate}Hz, got {audio_frame.sample_rate}Hz"
                    )

            # Put audio chunks into the queue
            await self._audio_queue.put(audio_data_float)

        except Exception as e:
            logger.error(f"Error feeding audio: {e}")

    def set_state(self, state: str):
        """
        Set the avatar's animation state.

        Args:
            state: One of 'idle', 'listening', 'thinking', 'speaking'
        """
        if state not in ["idle", "listening", "thinking", "speaking"]:
            logger.warning(f"Unknown state: {state}")
            return

        self.current_state = state
        logger.debug(f"Avatar state: {state}")

    async def _run_audio_processing(self):
        """
        Main audio processing loop.
        Consumes audio from the queue and feeds it to the Ditto SDK.
        """
        sdk_audio_buffer = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
        idle_task = None

        try:
            while self._running:
                # In idle/listening/thinking states, generate silent audio
                if self.current_state in ["idle", "listening", "thinking"]:
                    logger.debug(
                        "State is idle, listening, thinking --> creating silent frames"
                    )
                    if idle_task is None:
                        idle_task = asyncio.create_task(self._idle_audio_generator())

                    try:
                        # Try to get audio with timeout (in case TTS starts)
                        audio_chunk = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=0.2
                        )

                        # Got TTS audio, cancel idle generation
                        if idle_task:
                            idle_task.cancel()
                            idle_task = None

                        sdk_audio_buffer = np.concatenate(
                            [sdk_audio_buffer, audio_chunk]
                        )

                    except asyncio.TimeoutError:
                        # No TTS audio, use silent chunk
                        sdk_audio_buffer = np.concatenate(
                            [sdk_audio_buffer, self.silent_chunk]
                        )

                else:  # speaking state
                    logger.debug("State is speaking --> sending TTS chunks")
                    # Cancel idle task if running
                    if idle_task:
                        idle_task.cancel()
                        idle_task = None

                    # Get audio from queue (TTS output)
                    try:
                        audio_chunk = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=0.5
                        )
                        sdk_audio_buffer = np.concatenate(
                            [sdk_audio_buffer, audio_chunk]
                        )
                    except asyncio.TimeoutError:
                        # No audio available, use silent chunk
                        sdk_audio_buffer = np.concatenate(
                            [sdk_audio_buffer, self.silent_chunk]
                        )

                # Process buffered audio
                while len(sdk_audio_buffer) >= self.split_len:
                    audio_for_sdk = sdk_audio_buffer[: self.split_len]
                    sdk_audio_buffer = sdk_audio_buffer[self.chunksize[1] * 640 :]

                    # Push audio to AVSynchronizer for sync
                    # (This ensures all audio - TTS and silent - is synchronized with video)
                    try:
                        # Initialize base timestamp on first audio chunk
                        if self._base_timestamp is None:
                            self._base_timestamp = time.time()

                        # Calculate timestamp based on audio samples processed
                        # This ensures audio and video timestamps are perfectly aligned
                        audio_timestamp = self._base_timestamp + (
                            self._audio_samples_processed / self.sample_rate
                        )

                        # Convert float32 audio to int16 for AudioFrame
                        audio_int16 = (audio_for_sdk * 32768).astype(np.int16)

                        # Create AudioFrame
                        audio_frame = rtc.AudioFrame(
                            data=audio_int16.tobytes(),
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(audio_int16),
                        )

                        # Push to AVSynchronizer (non-blocking to avoid blocking Ditto pipeline)
                        future = asyncio.run_coroutine_threadsafe(
                            self.av_sync.push(audio_frame, audio_timestamp), self._loop
                        )
                        # Don't wait for completion to avoid blocking

                        # Update sample counter
                        self._audio_samples_processed += len(audio_for_sdk)

                    except Exception as e:
                        logger.warning(f"Error pushing audio to AVSynchronizer: {e}")

                    # Feed to Ditto SDK
                    async with self._sdk_lock:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.sdk.run_chunk, audio_for_sdk, self.chunksize
                        )

        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled")
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}", exc_info=True)
        finally:
            if idle_task:
                idle_task.cancel()

    async def _idle_audio_generator(self):
        """
        Generates silent audio chunks for idle animation.
        """
        IDLE_PERIOD_S = (self.chunksize[1] * 640) / self.sample_rate

        try:
            while True:
                await self._audio_queue.put(self.silent_chunk)
                await asyncio.sleep(IDLE_PERIOD_S)
        except asyncio.CancelledError:
            pass

    async def close(self):
        """Gracefully shuts down the worker and SDK."""
        logger.info("Shutting down avatar worker...")
        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Flush the SDK
        async with self._sdk_lock:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.sdk.run_chunk, self.silent_chunk, self.chunksize
                )
                await asyncio.get_event_loop().run_in_executor(None, self.sdk.close)
            except Exception as e:
                logger.error(f"Error closing SDK: {e}")

        logger.info("✅ Avatar worker shutdown complete")
