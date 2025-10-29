"""
Inference service that wraps StreamSDK for API use.
"""

import asyncio
import queue
import threading
import time
import base64
import io
from pathlib import Path
from typing import Optional, Dict, Callable, Any
import numpy as np
from PIL import Image

from stream_pipeline_online import StreamSDK
from core.utils.logging_config import get_logger, set_correlation_id, CorrelationIdContext
from api.config import settings

logger = get_logger(__name__)


class InferenceService:
    """
    Wraps StreamSDK for async API usage with streaming support.
    """

    def __init__(self, cfg_pkl: str, data_root: str):
        """
        Initialize the inference service.

        Args:
            cfg_pkl: Path to configuration pickle file
            data_root: Root directory for model data
        """
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.sdk: Optional[StreamSDK] = None
        self._lock = asyncio.Lock()

        logger.info(f"InferenceService initialized with backend config: {cfg_pkl}")

    async def initialize(self):
        """Initialize the StreamSDK instance."""
        async with self._lock:
            if self.sdk is None:
                logger.info("Initializing StreamSDK")
                start_time = time.time()

                # Run SDK initialization in thread executor to avoid blocking
                loop = asyncio.get_event_loop()
                self.sdk = await loop.run_in_executor(
                    None,
                    lambda: StreamSDK(self.cfg_pkl, self.data_root)
                )

                init_time = time.time() - start_time
                logger.info(f"StreamSDK initialized in {init_time:.2f}s")

    async def setup_session(
        self,
        session_id: str,
        source_path: str,
        output_path: str,
        **kwargs
    ) -> None:
        """
        Setup a streaming session.

        Args:
            session_id: Unique session identifier
            source_path: Path to source image/video
            output_path: Path for output video
            **kwargs: Additional configuration parameters
        """
        with CorrelationIdContext(session_id):
            logger.info(f"Setting up session {session_id}")

            # Ensure SDK is initialized
            if self.sdk is None:
                await self.initialize()

            # Set online mode for streaming
            kwargs['online_mode'] = True

            # Run setup in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sdk.setup(source_path, output_path, **kwargs)
            )

            logger.info(f"Session {session_id} setup complete")

    async def process_audio_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        chunksize: tuple = (3, 5, 2)
    ) -> None:
        """
        Process an audio chunk.

        Args:
            session_id: Session identifier
            audio_chunk: Audio data as numpy array (16kHz, float32)
            chunksize: Chunk size configuration for Hubert
        """
        with CorrelationIdContext(session_id):
            if self.sdk is None:
                raise RuntimeError("SDK not initialized")

            # Run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sdk.run_chunk(audio_chunk, chunksize)
            )

            logger.debug(f"Audio chunk processed for session {session_id}")

    async def setup_Nd(
        self,
        session_id: str,
        N_d: int,
        fps: int,
        fade_in: int = -1,
        fade_out: int = -1
    ) -> None:
        """
        Setup frame count after SDK is initialized (like inference.py line 43).

        Args:
            session_id: Session identifier
            N_d: Number of frames to generate
            fps: Frames per second (for logging only)
            fade_in: Fade in frames (-1 to disable)
            fade_out: Fade out frames (-1 to disable)
        """
        with CorrelationIdContext(session_id):
            logger.info(f"Setting up N_d={N_d} frames at {fps} fps for session {session_id}")
            audio_duration = N_d / fps if fps > 0 else 0
            logger.info(f"Expected video duration: {audio_duration:.2f} seconds")

            if self.sdk is None:
                raise RuntimeError("SDK not initialized")

            # Run setup_Nd in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sdk.setup_Nd(
                    N_d=N_d,
                    fade_in=fade_in,
                    fade_out=fade_out,
                    ctrl_info={}
                )
            )

            logger.info(f"N_d setup complete for session {session_id}")

    async def process_full_audio(
        self,
        session_id: str,
        audio: np.ndarray,
        chunksize: tuple = (3, 5, 2)
    ) -> None:
        """
        Process full audio file by splitting into chunks (like inference.py).

        Args:
            session_id: Session identifier
            audio: Full audio array (16kHz, float32)
            chunksize: Chunk size configuration (default: 3, 5, 2)
        """
        with CorrelationIdContext(session_id):
            logger.info(f"Processing full audio for session {session_id}, length: {len(audio)} samples")

            if self.sdk is None:
                raise RuntimeError("SDK not initialized")

            def process():
                # Add padding at the beginning (like inference.py line 48)
                audio_padded = np.concatenate([
                    np.zeros((chunksize[0] * 640,), dtype=np.float32),
                    audio
                ], 0)

                # Calculate split length (like inference.py line 49)
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480

                # Process in chunks (like inference.py lines 50-54)
                chunk_count = 0
                for i in range(0, len(audio_padded), chunksize[1] * 640):
                    audio_chunk = audio_padded[i:i + split_len]
                    if len(audio_chunk) < split_len:
                        audio_chunk = np.pad(
                            audio_chunk,
                            (0, split_len - len(audio_chunk)),
                            mode="constant"
                        )
                    self.sdk.run_chunk(audio_chunk, chunksize)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        logger.debug(f"Processed {chunk_count} chunks")

                logger.info(f"Processed {chunk_count} total chunks")

            # Run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, process)

            logger.info(f"Full audio processing complete for session {session_id}")

    async def close_session(self, session_id: str) -> None:
        """
        Close a session and finalize output.

        Args:
            session_id: Session identifier
        """
        with CorrelationIdContext(session_id):
            logger.info(f"Closing session {session_id}")

            if self.sdk is None:
                logger.warning("SDK not initialized, nothing to close")
                return

            # Run close in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.sdk.close
            )

            logger.info(f"Session {session_id} closed successfully")

    def get_writer_queue(self) -> Optional[queue.Queue]:
        """
        Get the writer queue for frame monitoring.

        Returns:
            Writer queue or None
        """
        if self.sdk:
            return self.sdk.writer_queue
        return None

    async def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.sdk is not None


class StreamingInferenceService:
    """
    Specialized service for WebSocket streaming with frame callbacks.
    """

    def __init__(self, cfg_pkl: str, data_root: str):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.sessions: Dict[str, InferenceService] = {}

    async def create_session(self, session_id: str) -> InferenceService:
        """
        Create a new inference session.

        Args:
            session_id: Unique session identifier

        Returns:
            InferenceService instance
        """
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists")
            return self.sessions[session_id]

        service = InferenceService(self.cfg_pkl, self.data_root)
        await service.initialize()
        self.sessions[session_id] = service

        logger.info(f"Created inference session: {session_id}")
        return service

    async def get_session(self, session_id: str) -> Optional[InferenceService]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self.sessions:
            # Close the session first
            service = self.sessions[session_id]
            try:
                await service.close_session(session_id)
            except Exception as e:
                logger.exception(f"Error closing session {session_id}")

            del self.sessions[session_id]
            logger.info(f"Removed inference session: {session_id}")
            return True
        return False

    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)


# Global inference service instance
_inference_service: Optional[StreamingInferenceService] = None


def get_inference_service(cfg_pkl: str, data_root: str) -> StreamingInferenceService:
    """
    Get or create the global inference service instance.

    Args:
        cfg_pkl: Path to configuration pickle
        data_root: Root directory for models

    Returns:
        StreamingInferenceService instance
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = StreamingInferenceService(cfg_pkl, data_root)
        logger.info("Global inference service created")
    return _inference_service


# Utility functions for data conversion
def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def base64_to_audio(base64_str: str, dtype=np.float32) -> np.ndarray:
    """Convert base64 string to audio numpy array."""
    audio_bytes = base64.b64decode(base64_str)
    # Assume int16 PCM audio
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    # Convert to float32 in range [-1, 1]
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return audio_float


def audio_to_base64(audio: np.ndarray) -> str:
    """Convert audio numpy array to base64 string."""
    # Convert float32 to int16
    audio_int16 = (audio * 32768.0).astype(np.int16)
    return base64.b64encode(audio_int16.tobytes()).decode('utf-8')
