import asyncio
import av
import numpy as np
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    OutputImageRawFrame,
    StartFrame,
    EndFrame,
)


class H264EncoderProcessor(FrameProcessor):
    """
    Encodes RGB image frames to H.264 video using FFmpeg/libx264.

    This processor:
    - Receives OutputImageRawFrame (RGB)
    - Encodes to H.264 using av (PyAV)
    - Outputs encoded H.264 frames suitable for WebRTC

    Note: For WebRTC, we don't actually need to output encoded frames from Pipecat,
    as the WebRTC transport handles encoding internally. This processor is mainly
    for demonstration or if you want to handle encoding separately.
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 25,
        bitrate: str = "2M",
        preset: str = "ultrafast",
        **kwargs
    ):
        """
        Initialize H.264 encoder.

        Args:
            width: Video width (None = auto-detect from first frame)
            height: Video height (None = auto-detect from first frame)
            fps: Frames per second
            bitrate: Target bitrate (e.g., "2M", "1500k")
            preset: x264 preset (ultrafast, fast, medium, slow, etc.)
            **kwargs: Additional FrameProcessor arguments
        """
        super().__init__(**kwargs)

        self._width = width
        self._height = height
        self._fps = fps
        self._bitrate = bitrate
        self._preset = preset

        # Encoder (lazy initialization)
        self._codec: Optional[av.codec.CodecContext] = None
        self._frame_count = 0

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._initialize_encoder()
            await self.push_frame(frame, direction)

        elif isinstance(frame, OutputImageRawFrame):
            # Encode the frame
            await self._encode_frame(frame)
            # Push original frame downstream
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self._cleanup()
            await self.push_frame(frame, direction)

        else:
            # Pass through other frames
            await self.push_frame(frame, direction)

    async def _initialize_encoder(self):
        """Initialize the H.264 encoder."""
        if self._codec is not None:
            return

        # Run initialization in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_encoder_sync)

    def _init_encoder_sync(self, width=None, height=None):
        """Synchronous encoder initialization."""
        # Use provided dimensions or the ones set in __init__
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height

        if self._width is None or self._height is None:
            raise ValueError("Cannot initialize encoder without width and height")

        # Create H.264 encoder
        self._codec = av.CodecContext.create("libx264", "w")

        self._codec.width = self._width
        self._codec.height = self._height
        self._codec.pix_fmt = "yuv420p"
        self._codec.time_base = av.Fraction(1, self._fps)
        self._codec.framerate = self._fps

        # Bitrate
        if isinstance(self._bitrate, str):
            if self._bitrate.endswith("M"):
                bitrate_value = int(float(self._bitrate[:-1]) * 1_000_000)
            elif self._bitrate.endswith("k"):
                bitrate_value = int(float(self._bitrate[:-1]) * 1_000)
            else:
                bitrate_value = int(self._bitrate)
        else:
            bitrate_value = self._bitrate

        self._codec.bit_rate = bitrate_value

        # x264 specific options
        self._codec.options = {
            "preset": self._preset,
            "tune": "zerolatency",  # Important for real-time
            "profile": "baseline",  # Compatible with WebRTC
        }

        # Open encoder
        self._codec.open()

        print(f"[H264Encoder] Initialized: {self._width}x{self._height} @ {self._fps}fps, bitrate={self._bitrate}, preset={self._preset}")

    async def _encode_frame(self, frame: OutputImageRawFrame):
        """Encode a single frame."""
        # Auto-detect dimensions from first frame if not set
        if self._codec is None:
            if self._width is None or self._height is None:
                # Get dimensions from frame
                width, height = frame.size
                print(f"[H264Encoder] Auto-detected dimensions: {width}x{height}")
                await self._initialize_encoder()
                # Update dimensions
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._init_encoder_sync, width, height)
            return

        # Run encoding in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._encode_frame_sync, frame)

    def _encode_frame_sync(self, frame: OutputImageRawFrame):
        """Synchronous frame encoding."""
        # Convert bytes to numpy array
        if isinstance(frame.image, bytes):
            img_array = np.frombuffer(frame.image, dtype=np.uint8)
            img_array = img_array.reshape((frame.size[1], frame.size[0], 3))  # (H, W, 3)
        else:
            img_array = frame.image

        # Create av VideoFrame
        av_frame = av.VideoFrame.from_ndarray(img_array, format="rgb24")

        # Set PTS (presentation timestamp)
        av_frame.pts = self._frame_count
        av_frame.time_base = av.Fraction(1, self._fps)

        # Encode
        packets = self._codec.encode(av_frame)

        # For WebRTC, we would send these packets
        # For now, just increment frame count
        self._frame_count += 1

        # Log occasionally
        if self._frame_count % 100 == 0:
            print(f"[H264Encoder] Encoded {self._frame_count} frames")

    async def _cleanup(self):
        """Cleanup encoder resources."""
        if self._codec:
            # Flush encoder
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._flush_encoder)

            self._codec.close()
            self._codec = None

        print(f"[H264Encoder] Cleaned up. Total frames: {self._frame_count}")

    def _flush_encoder(self):
        """Flush remaining frames from encoder."""
        if self._codec:
            packets = self._codec.encode(None)
            # Process remaining packets if needed
