"""
Gemini Live API Processor for Pipecat framework.

Handles bidirectional streaming with Gemini 2.0 Flash Live model:
- User audio input → Gemini ASR → LLM → TTS → Audio output
- Supports full-duplex conversations with interruption handling
- Extracts emotion for avatar expression control
"""

import asyncio
import numpy as np
import os
from typing import Optional, Callable
import queue
import threading

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    StartFrame,
    EndFrame,
    TextFrame,
)

import google.genai as genai
from google.genai import types

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conversation_manager import ConversationManager, EmotionType


class GeminiLiveProcessor(FrameProcessor):
    """
    Pipecat processor for Gemini Live API integration.

    Features:
    - Real-time audio streaming to Gemini
    - ASR + LLM + TTS in a single API call
    - Full-duplex conversation support
    - Emotion detection from responses
    - Conversation context management
    - Interruption handling with VAD
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemini-2.0-flash-exp",
        voice_name: str = "Puck",  # Gemini voice options: Puck, Charon, Kore, Fenrir, Aoede
        enable_interruptions: bool = True,
        vad_threshold: float = 0.5,
        on_emotion_change: Optional[Callable[[EmotionType], None]] = None,
        **kwargs
    ):
        """
        Initialize Gemini Live processor.

        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Gemini model name
            voice_name: Voice for TTS output
            enable_interruptions: Allow user to interrupt avatar
            vad_threshold: Voice activity detection threshold
            on_emotion_change: Callback when emotion changes
            **kwargs: Additional Pipecat FrameProcessor options
        """
        super().__init__(**kwargs)

        # API configuration
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter.")

        self.model = model
        self.voice_name = voice_name
        self.enable_interruptions = enable_interruptions
        self.vad_threshold = vad_threshold
        self.on_emotion_change = on_emotion_change

        # Gemini client and session
        self.client = genai.Client(api_key=self.api_key)
        self.session: Optional[Any] = None
        self._session_context = None  # Store the context manager

        # Conversation manager
        self.conversation = ConversationManager(
            max_history=20,
            context_window=10,
            emotion_smoothing=0.7
        )

        # Audio configuration
        self.input_sample_rate = 16000  # Gemini expects 16kHz
        self.output_sample_rate = 16000  # Gemini outputs 16kHz

        # State management
        self.is_speaking = False  # Avatar is speaking
        self.is_user_speaking = False  # User is speaking
        self._initialized = False
        self._running = False

        # Audio buffers
        self.input_buffer = []
        self.output_queue = asyncio.Queue()

        # Background tasks
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

        # Threading
        self._lock = asyncio.Lock()

        print(f"[GeminiLiveProcessor] Initialized with model: {model}, voice: {voice_name}")

    async def process_frame(self, frame: Frame, direction):
        """
        Process incoming frames.

        Audio handling strategy:
        1. User audio is ALWAYS sent to Gemini for ASR/conversation
        2. When avatar is NOT speaking (user speaking or silence):
           - Push SILENCE frames to DittoProcessor
           - Keeps avatar animated in idle/listening pose
           - Ditto's IdleAnimationProcessor handles subtle movements
        3. When avatar IS speaking:
           - Push Gemini's TTS audio to DittoProcessor
           - Avatar lip-syncs to Gemini's speech

        This ensures the avatar is ALWAYS receiving audio (silence or speech)
        so it never freezes and maintains natural idle animations.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._initialize_session()
            await self.push_frame(frame, direction)

        elif isinstance(frame, AudioRawFrame):
            # User audio input
            await self._process_user_audio(frame)

            # Push silence to Ditto when avatar is NOT speaking
            # This keeps the avatar animated (idle/listening pose) even when quiet
            if not self.is_speaking:
                # Create silence frame with same format as input
                silence = b'\x00' * len(frame.audio)
                silence_frame = AudioRawFrame(
                    audio=silence,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels
                )
                await self.push_frame(silence_frame)
            # Note: When avatar IS speaking, Gemini audio output will be pushed via _push_output_audio()

        elif isinstance(frame, EndFrame):
            await self._cleanup()
            await self.push_frame(frame, direction)

        else:
            # Pass through other frames
            await self.push_frame(frame, direction)

        # Push any Gemini audio output frames
        await self._push_output_audio()

    async def _initialize_session(self):
        """Initialize Gemini Live session."""
        if self._initialized:
            return

        try:
            # Configure Gemini Live session
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],  # We want audio output
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name
                        )
                    )
                ),
            )

            # Create live session
            print(f"[GeminiLiveProcessor] Connecting to Gemini Live API...")
            self._session_context = self.client.aio.live.connect(model=self.model, config=config)
            self.session = await self._session_context.__aenter__()

            # Send system instructions as initial message
            system_prompt = self.conversation._get_system_prompt()
            await self.session.send(
                input=system_prompt,
                end_of_turn=True
            )

            # Start background tasks
            self._running = True
            self._receive_task = asyncio.create_task(self._receive_loop())

            self._initialized = True
            print(f"[GeminiLiveProcessor] Session initialized successfully")

        except Exception as e:
            print(f"[GeminiLiveProcessor] Error initializing session: {e}")
            raise

    async def _process_user_audio(self, frame: AudioRawFrame):
        """Process user audio input and send to Gemini."""
        if not self._initialized or not self.session:
            return

        try:
            # Convert audio frame to numpy array
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if needed
            if frame.sample_rate != self.input_sample_rate:
                ratio = frame.sample_rate / self.input_sample_rate
                if ratio != 1.0:
                    from scipy import signal
                    num_samples = int(len(audio_data) / ratio)
                    audio_data = signal.resample(audio_data, num_samples)

            # Convert to int16 PCM for Gemini
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Simple VAD: check if user is speaking
            audio_level = np.abs(audio_data).mean()
            was_speaking = self.is_user_speaking
            self.is_user_speaking = audio_level > self.vad_threshold

            # Handle interruptions
            if self.enable_interruptions and self.is_user_speaking and self.is_speaking:
                # User interrupted avatar - stop avatar speech
                print("[GeminiLiveProcessor] User interruption detected")
                self.is_speaking = False
                # Could send interrupt signal to Gemini here

            # Send audio to Gemini
            if self.is_user_speaking or was_speaking:
                audio_bytes = audio_int16.tobytes()
                await self.session.send(data=audio_bytes, mime_type="audio/pcm")

                # Mark end of turn when user stops speaking
                if was_speaking and not self.is_user_speaking:
                    await self.session.send(end_of_turn=True)

        except Exception as e:
            print(f"[GeminiLiveProcessor] Error processing user audio: {e}")

    async def _receive_loop(self):
        """Background task to receive responses from Gemini."""
        try:
            while self._running:
                try:
                    # Receive from Gemini
                    async for response in self.session.receive():
                        await self._handle_gemini_response(response)

                except Exception as e:
                    print(f"[GeminiLiveProcessor] Error in receive loop: {e}")
                    if self._running:
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        break

        except asyncio.CancelledError:
            print("[GeminiLiveProcessor] Receive loop cancelled")
        except Exception as e:
            print(f"[GeminiLiveProcessor] Fatal error in receive loop: {e}")

    async def _handle_gemini_response(self, response):
        """Handle response from Gemini."""
        try:
            # Handle text response (for transcript/debugging)
            if hasattr(response, 'text') and response.text:
                print(f"[Gemini]: {response.text}")

                # Add to conversation history
                emotion = self.conversation._detect_emotion(response.text)
                await self.conversation.add_turn(
                    role="model",
                    text=response.text,
                    emotion=emotion
                )

                # Notify emotion change
                current_emotion = await self.conversation.get_current_emotion()
                if self.on_emotion_change:
                    self.on_emotion_change(current_emotion)

                # Push text frame for logging
                text_frame = TextFrame(text=response.text)
                await self.push_frame(text_frame)

            # Handle audio response
            if hasattr(response, 'data') and response.data:
                # response.data is the audio bytes (PCM16 at 16kHz)
                audio_data = response.data

                # Mark that avatar is speaking
                self.is_speaking = True

                # Queue audio for output
                await self.output_queue.put(audio_data)

            # Handle end of turn
            if hasattr(response, 'server_content') and response.server_content:
                if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                    self.is_speaking = False
                    print("[GeminiLiveProcessor] Turn complete")

        except Exception as e:
            print(f"[GeminiLiveProcessor] Error handling Gemini response: {e}")

    async def _push_output_audio(self):
        """Push Gemini audio output as Pipecat frames."""
        while not self.output_queue.empty():
            try:
                audio_bytes = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=0.001
                )

                # Create AudioRawFrame for output
                output_frame = AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.output_sample_rate,
                    num_channels=1
                )

                # Push downstream (to DittoProcessor)
                await self.push_frame(output_frame)

            except asyncio.TimeoutError:
                break
            except Exception as e:
                print(f"[GeminiLiveProcessor] Error pushing output audio: {e}")
                break

    async def _cleanup(self):
        """Cleanup resources."""
        print("[GeminiLiveProcessor] Cleaning up...")
        self._running = False

        # Cancel background tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close Gemini session
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"[GeminiLiveProcessor] Error closing session: {e}")
            self.session = None
            self._session_context = None

        self._initialized = False
        print("[GeminiLiveProcessor] Cleanup complete")

    async def send_text(self, text: str):
        """Send text message to Gemini (for testing/debugging)."""
        if self.session:
            await self.session.send(input=text, end_of_turn=True)

    async def get_conversation_summary(self):
        """Get conversation summary."""
        return await self.conversation.get_summary()
