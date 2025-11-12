import asyncio
import numpy as np
import queue
import threading
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    EndFrame,
)

from stream_pipeline_online import StreamSDK


class DittoAvatarProcessor(FrameProcessor):
    """
    Pipecat processor that wraps the Ditto StreamSDK for real-time talking head generation.

    This processor:
    - Receives AudioRawFrame from WebRTC audio input
    - Processes audio through Ditto pipeline (Audio2Motion → MotionStitch → Warp → Decode → PutBack)
    - Outputs OutputImageRawFrame for WebRTC video stream
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        **kwargs
    ):
        """
        Initialize the Ditto avatar processor.

        Args:
            cfg_pkl: Path to configuration pickle file
            data_root: Root directory for model data
            source_path: Path to source avatar image or video
            **kwargs: Additional StreamSDK configuration options
        """
        super().__init__(**kwargs)

        self._cfg_pkl = cfg_pkl
        self._data_root = data_root
        self._source_path = source_path
        self._sdk_kwargs = kwargs

        # StreamSDK instance (initialized on first frame)
        self._sdk: Optional[StreamSDK] = None
        self._initialized = False

        # Audio buffering
        self._audio_buffer = []
        self._audio_sample_rate = 16000  # Ditto expects 16kHz
        self._audio_channels = 1  # Mono

        # Frame output queue
        self._output_queue = asyncio.Queue()

        # Threading lock
        self._lock = threading.Lock()

        # Emotion control (for dynamic updates)
        self._current_emotion = kwargs.get('emo', 4)  # Default neutral

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Initialize StreamSDK on start
            await self._initialize_sdk()
            await self.push_frame(frame, direction)

        elif isinstance(frame, AudioRawFrame):
            # Process audio frame
            await self._process_audio(frame)
            # Push frame downstream (don't block the audio)
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            # Cleanup
            await self._cleanup()
            await self.push_frame(frame, direction)

        else:
            # Pass through other frames
            await self.push_frame(frame, direction)

        # Check if we have any output frames ready
        await self._push_output_frames()

    async def _initialize_sdk(self):
        """Initialize the StreamSDK (avatar registration and setup)."""
        if self._initialized:
            return

        # Run initialization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sdk_sync)

        self._initialized = True

    def _init_sdk_sync(self):
        """Synchronous SDK initialization."""
        # Create StreamSDK
        self._sdk = StreamSDK(
            self._cfg_pkl,
            self._data_root,
            **self._sdk_kwargs
        )

        # Setup with frame callback
        setup_kwargs = {
            "online_mode": True,  # Enable streaming mode
            "N_d": -1,  # Unknown number of frames
        }
        setup_kwargs.update(self._sdk_kwargs)

        self._sdk.setup(
            self._source_path,
            output_path=None,  # No file output
            frame_callback=self._on_frame_ready,
            **setup_kwargs
        )

        print(f"[DittoAvatarProcessor] Initialized with source: {self._source_path}")

    def _on_frame_ready(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """
        Callback from StreamSDK when a frame is ready.

        Args:
            frame_rgb: RGB frame array (H, W, 3) uint8
            frame_idx: Frame index
            timestamp: Frame timestamp in seconds
        """
        # Put frame in output queue (thread-safe)
        asyncio.run_coroutine_threadsafe(
            self._output_queue.put((frame_rgb, frame_idx, timestamp)),
            asyncio.get_event_loop()
        )

    async def _process_audio(self, frame: AudioRawFrame):
        """Process audio frame through StreamSDK."""
        if not self._initialized or self._sdk is None:
            return

        # Convert audio frame to numpy array
        # AudioRawFrame.audio is bytes, need to convert to float32
        audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample if needed (AudioRawFrame might be 48kHz, we need 16kHz)
        # For now, assume it's already 16kHz or handle resampling
        if frame.sample_rate != self._audio_sample_rate:
            # Simple decimation for now (proper resampling should use scipy.signal.resample)
            ratio = frame.sample_rate // self._audio_sample_rate
            if ratio > 1:
                audio_data = audio_data[::ratio]

        # Run audio processing in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sdk.run_chunk,
            audio_data,
            (3, 5, 2)  # HuBERT chunk size
        )

    async def _push_output_frames(self):
        """Push any ready output frames downstream."""
        while not self._output_queue.empty():
            try:
                frame_rgb, frame_idx, timestamp = await asyncio.wait_for(
                    self._output_queue.get(),
                    timeout=0.001
                )

                # Create Pipecat OutputImageRawFrame
                # frame_rgb is (H, W, 3) uint8 numpy array
                output_frame = OutputImageRawFrame(
                    image=frame_rgb.tobytes(),
                    size=(frame_rgb.shape[1], frame_rgb.shape[0]),  # (width, height)
                    format="RGB"
                )

                # Push to output
                await self.push_frame(output_frame)

            except asyncio.TimeoutError:
                break

    async def _cleanup(self):
        """Cleanup resources."""
        if self._sdk:
            # Run cleanup in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sdk.close)
            self._sdk = None

        self._initialized = False
        print("[DittoAvatarProcessor] Cleaned up")

    def update_emotion(self, emotion_code):
        """
        Update avatar emotion dynamically.

        Args:
            emotion_code: Emotion code string or int (0-7 or 'neu', 'hap', 'sad', 'ang', 'sur')

        Emotion mapping:
            0 or 'hap': Happy
            1 or 'ang': Angry
            2 or 'sad': Sad
            3: Fear
            4 or 'neu': Neutral
            5 or 'sur': Surprised
            6: Disgusted
            7: Contemptuous
        """
        # Map string codes to integers
        emotion_map = {
            'neu': 4,
            'hap': 0,
            'sad': 2,
            'ang': 1,
            'sur': 5,
        }

        if isinstance(emotion_code, str):
            emotion_code = emotion_map.get(emotion_code, 4)

        self._current_emotion = emotion_code

        # Update SDK emotion if initialized
        if self._sdk and hasattr(self._sdk, 'emo'):
            self._sdk.emo = emotion_code
            print(f"[DittoAvatarProcessor] Emotion updated to: {emotion_code}")
