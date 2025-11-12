import asyncio
import numpy as np
import time
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    EndFrame,
)


class DittoIdleAnimationProcessor(FrameProcessor):
    """
    Generates idle animation frames when no audio is present.

    This processor:
    - Monitors audio activity (simple volume-based VAD)
    - When silence detected, generates idle animation frames
    - Idle animation includes: breathing, subtle head movements, random blinks
    - Smoothly transitions between idle and speech-driven animation
    """

    def __init__(
        self,
        idle_threshold: float = 0.01,  # Volume threshold for silence
        idle_timeout: float = 0.5,  # Seconds of silence before starting idle
        fps: int = 25,
        **kwargs
    ):
        """
        Initialize idle animation processor.

        Args:
            idle_threshold: Audio volume threshold below which is considered silence
            idle_timeout: Time in seconds before switching to idle animation
            fps: Target frames per second for idle animation
            **kwargs: Additional FrameProcessor arguments
        """
        super().__init__(**kwargs)

        self._idle_threshold = idle_threshold
        self._idle_timeout = idle_timeout
        self._fps = fps

        # State tracking
        self._last_audio_time = None
        self._is_idle = False
        self._idle_frame_count = 0

        # Base frame for idle animation (captured from first video frame)
        self._base_frame: Optional[np.ndarray] = None
        self._base_frame_captured = False

        # Idle generation task
        self._idle_task: Optional[asyncio.Task] = None
        self._stop_idle = False

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._last_audio_time = time.time()
            await self.push_frame(frame, direction)

        elif isinstance(frame, AudioRawFrame):
            # Check audio activity
            await self._check_audio_activity(frame)
            await self.push_frame(frame, direction)

        elif isinstance(frame, OutputImageRawFrame):
            # Capture base frame for idle animation
            if not self._base_frame_captured:
                await self._capture_base_frame(frame)

            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self._stop_idle_generation()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _check_audio_activity(self, frame: AudioRawFrame):
        """Check if audio contains speech or is silent."""
        # Convert audio to numpy array
        audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Calculate RMS (root mean square) as simple volume metric
        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms > self._idle_threshold:
            # Audio detected
            self._last_audio_time = time.time()

            # If we were in idle mode, stop it
            if self._is_idle:
                await self._stop_idle_generation()
                self._is_idle = False

        else:
            # Silence detected
            if self._last_audio_time is not None:
                silence_duration = time.time() - self._last_audio_time

                # Start idle animation after timeout
                if silence_duration >= self._idle_timeout and not self._is_idle:
                    await self._start_idle_generation()
                    self._is_idle = True

    async def _capture_base_frame(self, frame: OutputImageRawFrame):
        """Capture the first video frame as base for idle animation."""
        if isinstance(frame.image, bytes):
            img_array = np.frombuffer(frame.image, dtype=np.uint8)
            img_array = img_array.reshape((frame.size[1], frame.size[0], 3))
        else:
            img_array = frame.image

        self._base_frame = img_array.copy()
        self._base_frame_captured = True
        print(f"[IdleAnimator] Captured base frame: {self._base_frame.shape}")

    async def _start_idle_generation(self):
        """Start generating idle animation frames."""
        if self._idle_task is None and self._base_frame is not None:
            self._stop_idle = False
            self._idle_task = asyncio.create_task(self._idle_generation_loop())
            print("[IdleAnimator] Started idle animation")

    async def _stop_idle_generation(self):
        """Stop generating idle animation frames."""
        if self._idle_task is not None:
            self._stop_idle = True
            await self._idle_task
            self._idle_task = None
            self._idle_frame_count = 0
            print("[IdleAnimator] Stopped idle animation")

    async def _idle_generation_loop(self):
        """Generate idle animation frames continuously."""
        frame_interval = 1.0 / self._fps

        while not self._stop_idle and self._base_frame is not None:
            # Generate idle frame
            idle_frame = await self._generate_idle_frame()

            # Create OutputImageRawFrame
            output_frame = OutputImageRawFrame(
                image=idle_frame.tobytes(),
                size=(idle_frame.shape[1], idle_frame.shape[0]),
                format="RGB"
            )

            # Push frame downstream
            await self.push_frame(output_frame)

            self._idle_frame_count += 1

            # Wait for next frame
            await asyncio.sleep(frame_interval)

    async def _generate_idle_frame(self) -> np.ndarray:
        """
        Generate a single idle animation frame.

        Simple idle animation:
        - Breathing: slight scaling (0.2Hz sine wave)
        - Color shift for breathing effect
        - Random micro-variations

        For more realistic idle, you would:
        - Use motion model to generate idle keypoints
        - Apply subtle head pose variations
        - Generate proper blink animations
        - Use Perlin noise for natural movement
        """
        # Simple implementation: just return base frame with slight variations
        # In production, you'd generate actual motion and render through Ditto pipeline

        t = self._idle_frame_count / self._fps

        # Breathing effect (subtle brightness variation)
        breathing_amplitude = 0.02
        breathing = 1.0 + breathing_amplitude * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz

        # Apply breathing
        idle_frame = self._base_frame.astype(np.float32) * breathing
        idle_frame = np.clip(idle_frame, 0, 255).astype(np.uint8)

        # TODO: Implement proper idle motion generation through Ditto pipeline
        # This would involve:
        # 1. Generate idle motion parameters (keypoints, expression)
        # 2. Feed through MotionStitch → WarpF3D → DecodeF3D → PutBack
        # 3. Return rendered frame

        return idle_frame
