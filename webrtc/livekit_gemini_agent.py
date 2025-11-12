"""
LiveKit + Gemini + Ditto Conversational Avatar Agent

This agent combines:
- LiveKit for WebRTC (audio/video streaming)
- Gemini Live API for conversation (ASR + LLM + TTS with built-in VAD and turn detection)
- Ditto for avatar video generation (always running at 25 FPS)

Architecture:
    - Gemini handles VAD and turn detection automatically
    - IDLE: Avatar in idle animation (silent audio to Ditto)
    - SPEAKING: AI responding (Gemini TTS audio to Ditto)

The avatar is ALWAYS visible and generating frames - state determines audio routing.
Gemini automatically detects when user speaks and handles turn-taking.

Usage:
    export LIVEKIT_URL=ws://localhost:7880
    export LIVEKIT_API_KEY=devkey
    export LIVEKIT_API_SECRET=devsecret
    export GEMINI_API_KEY=your-api-key

    python livekit_gemini_agent.py dev
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from enum import Enum
from collections import deque
import numpy as np
import websockets.exceptions

# LiveKit
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
# Note: VAD is handled by Gemini Live API - no need for silero

# Gemini
from google import genai
from google.genai import types

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stream_pipeline_online import StreamSDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Control websockets library logging (used by Gemini SDK)
# Set to WARNING to reduce noise from WebSocket message logs
websockets_logger = logging.getLogger('websockets')
websockets_log_level = os.getenv('WEBSOCKETS_LOG_LEVEL', 'WARNING').upper()
websockets_logger.setLevel(getattr(logging, websockets_log_level, logging.WARNING))
logger.info(f"🔇 WebSocket logging level: {websockets_log_level} (set WEBSOCKETS_LOG_LEVEL=DEBUG to see all messages)")


class ConversationState(Enum):
    """Conversation state - simplified since Gemini handles turn detection."""
    IDLE = "idle"              # No AI speech happening
    SPEAKING = "speaking"      # AI is responding (Gemini TTS audio)


class ConversationStateManager:
    """
    Manages conversation state - simplified since Gemini handles VAD and turn detection.

    State Machine (simplified):
        IDLE → SPEAKING (when Gemini starts responding)
        SPEAKING → IDLE (when Gemini finishes responding)

    Gemini automatically:
    - Detects when user starts/stops speaking (built-in VAD)
    - Processes user audio when speech ends
    - Handles turn-taking and interruptions
    - Manages conversation flow

    This manager only tracks if AI is speaking to route audio to Ditto correctly.
    """

    def __init__(self):
        self.state = ConversationState.IDLE
        self.lock = asyncio.Lock()

        # Silent audio generation (for IDLE state)
        # Chunk size formula: int(sum(chunksize) * 0.04 * 16000) + 80
        # Using default chunksize=(3,5,2) → 6480 samples @ 16kHz
        # Each chunk generates 5 frames (chunksize[1]=5)
        # For 25 FPS: need 5 Hz feed rate → 200ms sleep interval
        # TensorRT valid range: [3240..12960] samples
        self.silent_audio_chunk_size = 6480  # Default chunk size
        self.silent_audio_chunksize = (3, 5, 2)  # Corresponding chunksize parameter
        self.silent_audio = np.zeros(self.silent_audio_chunk_size, dtype=np.float32)

    async def transition_to(self, new_state: ConversationState):
        """Thread-safe state transition."""
        async with self.lock:
            old_state = self.state
            self.state = new_state
            logger.info(f"🔄 State transition: {old_state.value} → {new_state.value}")

    def get_state(self) -> ConversationState:
        """Get current state."""
        return self.state

    def is_speaking(self) -> bool:
        """Check if AI is currently speaking."""
        return self.state == ConversationState.SPEAKING

    def get_silent_audio(self) -> tuple[np.ndarray, tuple]:
        """Get silent audio chunk and chunksize for IDLE state."""
        return self.silent_audio.copy(), self.silent_audio_chunksize


# Log authentication method at module load time
def _log_auth_config():
    """Log which authentication method is configured."""
    vertex_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    vertex_project = os.getenv("VERTEX_PROJECT_ID")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if vertex_creds:
        logger.info("=" * 60)
        logger.info("🔐 Vertex AI Authentication Configured")
        logger.info(f"   Credentials: {vertex_creds}")
        logger.info(f"   Project ID:  {vertex_project or 'NOT SET ⚠️'}")
        logger.info(f"   Location:    {os.getenv('VERTEX_LOCATION', 'us-central1')}")
        if not vertex_project:
            logger.warning("⚠️  VERTEX_PROJECT_ID not set - authentication may fail!")
        logger.info("=" * 60)
    elif gemini_api_key:
        logger.info("=" * 60)
        logger.info("🔑 API Key Authentication Configured")
        logger.info(f"   API Key: {gemini_api_key[:10]}...")
        logger.info("=" * 60)
    else:
        logger.warning("⚠️  No authentication method configured!")

_log_auth_config()


class GeminiDittoAgent:
    """
    Conversational avatar agent using Gemini Live API + Ditto.

    Flow:
    1. User audio → Gemini Live API
    2. Gemini responds with audio
    3. Gemini audio → Ditto model
    4. Ditto generates video frames
    5. Video frames → LiveKit → Browser
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-live-2.5-flash-preview-native-audio-09-2025",
        voice_name: str = "Puck",
        system_instruction: str = None,
        **ditto_kwargs
    ):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.source_path = source_path
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.voice_name = voice_name
        self.system_instruction = system_instruction or "You are a helpful AI assistant."
        self.ditto_kwargs = ditto_kwargs

        # Components
        self.sdk: Optional[StreamSDK] = None
        self.gemini_client = None
        self.gemini_session = None

        # LiveKit
        self.room: Optional[rtc.Room] = None
        self.video_source: Optional[rtc.VideoSource] = None
        self.audio_source: Optional[rtc.AudioSource] = None

        # Conversation state management
        self.state_manager = ConversationStateManager()

        # Note: VAD is handled by Gemini Live API - no custom VAD needed

        # Audio buffers
        self.gemini_audio_buffer = np.array([], dtype=np.float32)
        # Universal chunk size using default chunksize=(3,5,2)
        # Formula: int(sum(chunksize) * 0.04 * 16000) + 80 = 6480 samples
        # 6480 samples @ 16kHz = 405ms → generates ~5 frames per chunk
        # Valid range: [3240..12960] samples (TensorRT optimization profile)
        self.model_chunk_size = 6480  # Default chunk size for chunksize=(3,5,2)
        self.model_chunksize = (3, 5, 2)  # Chunksize parameter for StreamSDK

        # Frame timing
        self.target_fps = 60  # Send frames at 60 FPS max
        self.last_frame_timestamp = None  # Last Ditto timestamp (in video timeline)
        self.frame_interval = 1.0 / self.target_fps  # 16.67ms for 60 FPS

        # Silent audio generation task
        self.silent_audio_task: Optional[asyncio.Task] = None

        # Track last time we fed audio to Ditto (to prevent starvation)
        self.last_ditto_feed_time = time.time()
        self.ditto_feed_timeout = 0.15  # 150ms max gap before feeding silent audio

        # Lock to serialize Ditto SDK access (TensorRT is NOT thread-safe)
        self.ditto_lock = asyncio.Lock()

        # Statistics
        self._frames_generated = 0
        self._frames_dropped = 0
        self._gemini_chunks_received = 0
        self._ditto_chunks_sent = 0

        # Track if using Vertex AI
        self._using_vertex_ai = False

        # Profiling: Track timing for each stage
        self._profile_timings = {
            'user_audio_processing': deque(maxlen=100),
            'gemini_send': deque(maxlen=100),
            'gemini_response': deque(maxlen=100),
            'gemini_audio_processing': deque(maxlen=100),
            'audio_resample': deque(maxlen=100),
            'ditto_chunk': deque(maxlen=100),
            'frame_generation': deque(maxlen=100),
            'silent_audio_gen': deque(maxlen=100),
        }
        self._enable_profiling = os.getenv("ENABLE_PROFILING", "true").lower() == "true"

        # Event loop reference (set during initialize)
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _profile_stage(self, stage_name: str, duration_ms: float, details: str = ""):
        """Log and track profiling information for a stage."""
        if not self._enable_profiling:
            return
        
        # Store timing
        if stage_name in self._profile_timings:
            self._profile_timings[stage_name].append(duration_ms)
            avg = sum(self._profile_timings[stage_name]) / len(self._profile_timings[stage_name])
            logger.debug(f"⏱️  [{stage_name}] {duration_ms:.2f}ms (avg: {avg:.2f}ms) {details}")
        else:
            logger.debug(f"⏱️  [{stage_name}] {duration_ms:.2f}ms {details}")

    def _get_profile_stats(self) -> dict:
        """Get profiling statistics summary."""
        stats = {}
        for stage, timings in self._profile_timings.items():
            if timings:
                stats[stage] = {
                    'count': len(timings),
                    'avg_ms': sum(timings) / len(timings),
                    'min_ms': min(timings),
                    'max_ms': max(timings),
                    'total_ms': sum(timings)
                }
        return stats

    def _format_model_name(self, model: str) -> str:
        """
        Format model name correctly for Vertex AI or API key authentication.
        
        For Vertex AI: Remove "models/" prefix (Vertex AI uses just the model name)
        For API key: Use the model name as-is (with models/ prefix if present)
        """
        if self._using_vertex_ai:
            # For Vertex AI, remove "models/" prefix if present
            # Vertex AI expects just the model name like "gemini-live-2.5-flash-preview-native-audio-09-2025"
            if model.startswith("models/"):
                model = model.replace("models/", "", 1)
            return model
        else:
            # For API key, keep the model name as-is (with models/ prefix)
            return model

    async def initialize(self):
        """Initialize Ditto SDK and Gemini client."""
        # Store event loop reference for frame capture from worker threads
        self._event_loop = asyncio.get_running_loop()
        logger.info("🔄 Event loop reference stored for frame capture")

        logger.info("🎭 Initializing Ditto SDK...")
        await asyncio.to_thread(self._init_sdk_sync)
        logger.info("✅ Ditto SDK initialized")
        logger.info("ℹ️  VAD and turn detection handled by Gemini Live API")
        if self._enable_profiling:
            logger.info("📊 Profiling enabled - set ENABLE_PROFILING=false to disable")

        logger.info("🤖 Initializing Gemini Live API...")

        # Check for Vertex AI credentials first (prioritize Vertex AI if both are set)
        vertex_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        vertex_project = os.getenv("VERTEX_PROJECT_ID")
        
        if vertex_creds:
            # Use Vertex AI with service account credentials
            self._using_vertex_ai = True
            logger.info("🔐 Using Vertex AI authentication")
            logger.info(f"   Credentials: {vertex_creds}")
            logger.info(f"   Project ID: {vertex_project or 'NOT SET (may fail)'}")
            logger.info(f"   Location: {os.getenv('VERTEX_LOCATION', 'us-central1')}")
            
            if not vertex_project:
                logger.warning("⚠️  VERTEX_PROJECT_ID not set - this may cause authentication to fail")
            
            # The google-genai library will automatically use GOOGLE_APPLICATION_CREDENTIALS
            self.gemini_client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=os.getenv("VERTEX_LOCATION", "us-central1")
            )
            
            # Format model name for Vertex AI
            original_model = self.gemini_model
            self.gemini_model = self._format_model_name(self.gemini_model)
            logger.info(f"   Model: {original_model} → {self.gemini_model}")
        elif self.gemini_api_key:
            # Use API key authentication
            self._using_vertex_ai = False
            logger.info("🔑 Using Gemini API key authentication")
            logger.info(f"   API Key: {self.gemini_api_key[:10]}...")
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            # Model name stays as-is for API key (with models/ prefix)
        else:
            raise ValueError(
                "No authentication method found. Set either:\n"
                "  - GOOGLE_APPLICATION_CREDENTIALS (for Vertex AI)\n"
                "  - GEMINI_API_KEY (for API key auth)"
            )

        logger.info("✅ Gemini client initialized")

    def _init_sdk_sync(self):
        """Synchronous SDK initialization."""
        logger.info("🔧 Initializing StreamSDK...")
        logger.info(f"   Config: {self.cfg_pkl}")
        logger.info(f"   Data root: {self.data_root}")
        logger.info(f"   Source: {self.source_path}")
        
        init_start = time.perf_counter()
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root, **self.ditto_kwargs)
        init_time = (time.perf_counter() - init_start) * 1000
        logger.info(f"✅ StreamSDK created ({init_time:.2f}ms)")

        setup_kwargs = {
            "online_mode": True,
            "N_d": -1,
        }
        setup_kwargs.update(self.ditto_kwargs)
        
        logger.info("🔧 Setting up StreamSDK pipeline...")
        logger.debug(f"   Setup kwargs: {setup_kwargs}")

        setup_start = time.perf_counter()
        self.sdk.setup(
            self.source_path,
            output_path=None,
            frame_callback=self._on_frame_generated,
            **setup_kwargs
        )
        setup_time = (time.perf_counter() - setup_start) * 1000
        logger.info(f"✅ StreamSDK setup complete ({setup_time:.2f}ms)")
        logger.info(f"   Online mode: {self.sdk.online_mode}")
        logger.info(f"   Streaming mode: {self.sdk.streaming_mode}")

    def _on_frame_generated(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """Callback when Ditto generates a frame (from worker thread)."""
        stage_start = time.perf_counter()
        self._frames_generated += 1

        if self._frames_generated == 1:
            logger.info(f"🎬 First frame generated from StreamSDK: {frame_rgb.shape}")
            logger.info(f"   Frame index: {frame_idx}, Timestamp: {timestamp:.3f}s")
            self.last_frame_timestamp = timestamp

        # Log every 100 frames for StreamSDK monitoring
        if self._frames_generated % 100 == 0:
            logger.debug(f"📹 StreamSDK: Generated {self._frames_generated} frames (latest idx: {frame_idx}, timestamp: {timestamp:.3f}s)")

        # Frame selection based on target FPS (60 FPS)
        # Only send frames that advance the video timeline by at least 16.67ms
        if self.last_frame_timestamp is not None:
            timestamp_delta = timestamp - self.last_frame_timestamp
            if timestamp_delta < self.frame_interval:
                # Frame is too close to previous frame in video timeline - skip it
                self._frames_dropped += 1
                logger.debug(f"⏭️  Skipping frame {frame_idx}: timestamp delta {timestamp_delta*1000:.1f}ms < {self.frame_interval*1000:.1f}ms target (60 FPS)")
                return

        self.last_frame_timestamp = timestamp

        if self.video_source:
            try:
                rgb_to_rgba_start = time.perf_counter()
                rgba_data = self._rgb_to_rgba(frame_rgb)
                rgb_to_rgba_time = (time.perf_counter() - rgb_to_rgba_start) * 1000
                self._profile_stage('frame_generation', rgb_to_rgba_time, f"RGB→RGBA conversion")

                video_frame = rtc.VideoFrame(
                    width=frame_rgb.shape[1],
                    height=frame_rgb.shape[0],
                    type=rtc.VideoBufferType.RGBA,
                    data=rgba_data
                )

                # Schedule frame for immediate transmission (no pacing delay)
                capture_start = time.perf_counter()
                asyncio.run_coroutine_threadsafe(
                    self._capture_frame_async(video_frame, stage_start),
                    self._event_loop
                )
                capture_time = (time.perf_counter() - capture_start) * 1000
                self._profile_stage('frame_generation', capture_time, f"Schedule frame capture")

                if self._frames_generated % 100 == 0:
                    logger.info(f"📊 Frames: {self._frames_generated} generated, {self._frames_dropped} skipped")
                    stats = self._get_profile_stats()
                    if 'frame_generation' in stats:
                        logger.info(f"📈 Frame generation stats: {stats['frame_generation']}")
                    # Log StreamSDK status periodically
                    self._log_streamsdk_status()

            except Exception as e:
                logger.error(f"Error preparing frame: {e}")

    async def _capture_frame_async(self, video_frame: rtc.VideoFrame, stage_start: float):
        """Capture frame immediately in asyncio context (no pacing delay)."""
        try:
            capture_start = time.perf_counter()
            self.video_source.capture_frame(video_frame)
            capture_time = (time.perf_counter() - capture_start) * 1000
            self._profile_stage('frame_generation', capture_time, f"LiveKit capture")

            total_time = (time.perf_counter() - stage_start) * 1000
            self._profile_stage('frame_generation', total_time, f"Total (frame #{self._frames_generated})")
        except Exception as e:
            logger.error(f"Error capturing frame in async context: {e}")

    def _rgb_to_rgba(self, rgb: np.ndarray) -> bytes:
        """Convert RGB to RGBA."""
        h, w, _ = rgb.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        return rgba.tobytes()

    async def start_gemini_session(self):
        """Start Gemini Live API session and keep it alive."""
        logger.info("🚀 Starting Gemini Live session...")
        logger.info(f"   Using model: {self.gemini_model}")

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],  # Get audio responses
            system_instruction=self.system_instruction,  # Set system instruction in config
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name
                    )
                )
            )
        )

        # Keep reconnecting if session drops
        while True:
            try:
                async with self.gemini_client.aio.live.connect(
                    model=self.gemini_model,
                    config=config
                ) as session:
                    self.gemini_session = session

                    logger.info("✅ Gemini Live session started")

                    # Process Gemini responses indefinitely
                    try:
                        logger.info("📡 Listening for Gemini responses...")
                        async for response in session.receive():
                            logger.debug(f"📨 Received response: {type(response)}")
                            await self._handle_gemini_response(response)
                        logger.warning("⚠️  Gemini receive loop ended - reconnecting...")
                    except asyncio.CancelledError:
                        logger.info("🛑 Gemini session cancelled")
                        raise
                    except Exception as e:
                        logger.error(f"❌ Error in Gemini receive loop: {e}")
                        # Will reconnect after a short delay
                    finally:
                        # Clear session reference when exiting context
                        self.gemini_session = None
                        logger.debug("🔌 Gemini session closed, cleared reference")

            except asyncio.CancelledError:
                logger.info("🛑 Gemini session task cancelled")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                error_msg = str(e)
                if "Permission" in error_msg or "denied" in error_msg:
                    logger.error("=" * 60)
                    logger.error("❌ PERMISSIONS ERROR: Service account lacks required IAM permissions")
                    logger.error("=" * 60)
                    raise
                # Otherwise, just log and reconnect
                logger.warning(f"⚠️  Connection closed: {e} - reconnecting in 2s...")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Error in Gemini session: {e} - reconnecting in 2s...")
                await asyncio.sleep(2)

    async def _handle_gemini_response(self, response):
        """Handle responses from Gemini Live API with state management."""
        stage_start = time.perf_counter()
        if response.server_content:
            # Check if model_turn exists and has parts
            if response.server_content.model_turn and response.server_content.model_turn.parts:
                # Check for audio in response
                has_audio = False
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                        # Got audio from Gemini!
                        has_audio = True
                        audio_data = part.inline_data.data
                        
                        # Transition to SPEAKING state when we receive audio
                        if not self.state_manager.is_speaking():
                            await self.state_manager.transition_to(ConversationState.SPEAKING)
                        
                        await self._process_gemini_audio(audio_data)

                    elif part.text:
                        logger.info(f"💬 Gemini: {part.text}")

                # If no more audio is coming, transition back to IDLE
                if not has_audio and self.state_manager.is_speaking():
                    await self.state_manager.transition_to(ConversationState.IDLE)
        
        response_time = (time.perf_counter() - stage_start) * 1000
        self._profile_stage('gemini_response', response_time, f"Handle Gemini response")

    async def _process_gemini_audio(self, audio_bytes: bytes):
        """Process audio from Gemini and feed to Ditto (only when SPEAKING)."""
        stage_start = time.perf_counter()
        self._gemini_chunks_received += 1

        # Convert audio bytes to numpy array
        # Gemini returns 24kHz PCM16
        convert_start = time.perf_counter()
        audio_data_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_data = audio_data_int16.astype(np.float32) / 32768.0
        convert_time = (time.perf_counter() - convert_start) * 1000
        self._profile_stage('gemini_audio_processing', convert_time, f"Convert bytes→numpy ({len(audio_data)} samples)")

        # IMPORTANT: Send audio to browser so user can HEAR the avatar!
        if self.audio_source:
            # Resample 24kHz → 48kHz for LiveKit output
            import scipy.signal
            resample_start = time.perf_counter()
            audio_48khz = scipy.signal.resample_poly(audio_data, up=2, down=1)
            resample_time = (time.perf_counter() - resample_start) * 1000
            self._profile_stage('audio_resample', resample_time, f"24kHz→48kHz ({len(audio_data)}→{len(audio_48khz)} samples)")
            
            audio_48khz_int16 = (audio_48khz * 32768).astype(np.int16)

            # Create audio frame for LiveKit (48kHz, mono, 20ms chunks)
            samples_per_chunk = 48000 // 50  # 960 samples = 20ms @ 48kHz
            capture_start = time.perf_counter()
            for i in range(0, len(audio_48khz_int16), samples_per_chunk):
                chunk_48k = audio_48khz_int16[i:i+samples_per_chunk]
                if len(chunk_48k) == samples_per_chunk:
                    audio_frame = rtc.AudioFrame(
                        data=chunk_48k.tobytes(),
                        sample_rate=48000,
                        num_channels=1,
                        samples_per_channel=samples_per_chunk
                    )
                    await self.audio_source.capture_frame(audio_frame)
            capture_time = (time.perf_counter() - capture_start) * 1000
            self._profile_stage('gemini_audio_processing', capture_time, f"Send to browser ({len(audio_48khz_int16)} samples)")

        # Only route to Ditto when in SPEAKING state
        if self.state_manager.get_state() != ConversationState.SPEAKING:
            return

        # Resample 24kHz → 16kHz for Ditto
        import scipy.signal
        resample_start = time.perf_counter()
        audio_data_16k = scipy.signal.resample_poly(audio_data, up=2, down=3)
        resample_time = (time.perf_counter() - resample_start) * 1000
        self._profile_stage('audio_resample', resample_time, f"24kHz→16kHz ({len(audio_data)}→{len(audio_data_16k)} samples)")

        # Add to buffer
        buffer_start = time.perf_counter()
        self.gemini_audio_buffer = np.concatenate([self.gemini_audio_buffer, audio_data_16k])
        buffer_time = (time.perf_counter() - buffer_start) * 1000
        self._profile_stage('gemini_audio_processing', buffer_time, f"Buffer audio (buffer size: {len(self.gemini_audio_buffer)})")

        # Process complete chunks
        chunks_processed = 0
        while len(self.gemini_audio_buffer) >= self.model_chunk_size:
            chunk = self.gemini_audio_buffer[:self.model_chunk_size]
            self.gemini_audio_buffer = self.gemini_audio_buffer[self.model_chunk_size:]

            # Feed to Ditto (with lock to prevent race condition)
            ditto_start = time.perf_counter()
            logger.debug(f"🎤 StreamSDK: Feeding chunk to run_chunk ({len(chunk)} samples, chunksize={self.model_chunksize}, buffer size: {len(self.gemini_audio_buffer)})")
            async with self.ditto_lock:
                await asyncio.to_thread(
                    self.sdk.run_chunk,
                    chunk,
                    self.model_chunksize
                )
                # Update last feed time to prevent starvation
                self.last_ditto_feed_time = time.time()
            ditto_time = (time.perf_counter() - ditto_start) * 1000
            self._profile_stage('ditto_chunk', ditto_time, f"Ditto run_chunk ({len(chunk)} samples)")
            chunks_processed += 1
            self._ditto_chunks_sent += 1
            logger.debug(f"✅ StreamSDK: run_chunk completed ({ditto_time:.2f}ms)")
            
            # Log queue status every 10 chunks to monitor StreamSDK health
            if self._ditto_chunks_sent % 10 == 0:
                queue_stats = self._get_streamsdk_queue_stats()
                if queue_stats:
                    queue_str = ', '.join([f"{k}={v['size']}/{v['maxsize']}" for k, v in queue_stats.items()])
                    logger.debug(f"📊 StreamSDK queues: {queue_str}")

        total_time = (time.perf_counter() - stage_start) * 1000
        self._profile_stage('gemini_audio_processing', total_time, f"Total (chunk #{self._gemini_chunks_received}, processed {chunks_processed} Ditto chunks)")

        if self._gemini_chunks_received % 10 == 0:
            logger.debug(f"🎤 Processed {self._gemini_chunks_received} Gemini audio chunks")
            stats = self._get_profile_stats()
            if 'gemini_audio_processing' in stats:
                logger.info(f"📈 Gemini audio processing stats: {stats['gemini_audio_processing']}")
            if 'ditto_chunk' in stats:
                logger.info(f"📈 Ditto chunk stats: {stats['ditto_chunk']}")

    async def run_silent_audio_generator(self):
        """
        Continuously feed silent audio to Ditto to prevent starvation.

        Strategy:
        - IDLE state: Always feed silent audio every 200ms
        - SPEAKING state: Feed silent audio only if Ditto hasn't received audio for >150ms
          (prevents starvation during gaps in Gemini audio)

        Using default chunksize=(3,5,2):
        - Chunk size: 6480 samples @ 16kHz (405ms audio duration)
        - Generates 5 frames per chunk (chunksize[1]=5)
        - Feeding keeps GPU warm and video smooth
        """
        logger.info("🔇 Starting silent audio generator (prevents Ditto starvation)")
        logger.info(f"   Chunk size: {self.state_manager.silent_audio_chunk_size} samples")
        logger.info(f"   Chunksize: {self.state_manager.silent_audio_chunksize}")
        logger.info(f"   IDLE: Feed every 200ms")
        logger.info(f"   SPEAKING: Feed only if gap > 150ms (safety net)")

        while True:
            try:
                state = self.state_manager.get_state()
                current_time = time.time()
                time_since_last_feed = current_time - self.last_ditto_feed_time

                # Determine if we should feed silent audio
                should_feed = False
                reason = ""

                if state == ConversationState.IDLE:
                    # Always feed during IDLE
                    should_feed = True
                    reason = "IDLE state"
                elif time_since_last_feed > self.ditto_feed_timeout:
                    # Feed during SPEAKING if gap detected (Gemini audio gap)
                    should_feed = True
                    reason = f"Gap detected ({time_since_last_feed*1000:.0f}ms since last feed)"

                if should_feed:
                    stage_start = time.perf_counter()
                    silent_chunk, chunksize = self.state_manager.get_silent_audio()

                    # Feed silent audio to Ditto (with lock to prevent race condition)
                    ditto_start = time.perf_counter()
                    logger.debug(f"🔇 StreamSDK: Feeding silent audio ({reason})")
                    async with self.ditto_lock:
                        await asyncio.to_thread(
                            self.sdk.run_chunk,
                            silent_chunk,
                            chunksize
                        )
                        # Update last feed time
                        self.last_ditto_feed_time = time.time()
                    ditto_time = (time.perf_counter() - ditto_start) * 1000
                    self._profile_stage('ditto_chunk', ditto_time, f"Silent audio chunk")
                    logger.debug(f"✅ StreamSDK: Silent audio processed ({ditto_time:.2f}ms)")

                    total_time = (time.perf_counter() - stage_start) * 1000
                    self._profile_stage('silent_audio_gen', total_time, f"Total silent audio generation")

                # Check every 50ms for responsive gap detection
                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                logger.info("🛑 Silent audio generator stopped")
                break
            except Exception as e:
                logger.error(f"❌ Error in silent audio generator: {e}")
                await asyncio.sleep(0.1)

    async def send_user_audio_to_gemini(self, audio_frame: rtc.AudioFrame):
        """
        Forward user audio directly to Gemini.
        
        Gemini handles:
        - VAD (detects when user starts/stops speaking)
        - Turn detection (knows when to process and respond)
        - Interruptions (can handle user speaking while AI is speaking)
        
        We just forward the audio - no manual VAD or accumulation needed!
        """
        stage_start = time.perf_counter()
        if not self.gemini_session:
            return

        # Convert audio to PCM16 format for Gemini
        convert_start = time.perf_counter()
        if audio_frame.sample_rate == 48000:
            # Resample 48kHz → 16kHz for Gemini
            audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Handle stereo → mono
            if audio_frame.num_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)

            import scipy.signal
            resample_start = time.perf_counter()
            audio_data_16k = scipy.signal.resample_poly(audio_data, up=1, down=3)
            resample_time = (time.perf_counter() - resample_start) * 1000
            self._profile_stage('audio_resample', resample_time, f"48kHz→16kHz ({len(audio_data)}→{len(audio_data_16k)} samples)")
            
            audio_int16 = (audio_data_16k * 32768).astype(np.int16)
        else:
            audio_int16 = np.frombuffer(audio_frame.data, dtype=np.int16)
        
        convert_time = (time.perf_counter() - convert_start) * 1000
        self._profile_stage('user_audio_processing', convert_time, f"Audio conversion ({len(audio_int16)} samples)")

        # Forward audio directly to Gemini - it handles VAD and turn detection
        try:
            send_start = time.perf_counter()
            await self.gemini_session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            mime_type="audio/pcm",
                            data=audio_int16.tobytes()
                        )
                    ]
                )
            )
            send_time = (time.perf_counter() - send_start) * 1000
            self._profile_stage('gemini_send', send_time, f"Send to Gemini ({len(audio_int16)} samples)")
            
            total_time = (time.perf_counter() - stage_start) * 1000
            self._profile_stage('user_audio_processing', total_time, f"Total user audio processing")
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError) as e:
            # Connection closed - session will be reconnecting, just skip this audio chunk
            logger.debug(f"🔌 Connection closed while sending audio (will reconnect): {e.code if hasattr(e, 'code') else 'unknown'}")
            # Session reference will be cleared by reconnect logic
        except Exception as e:
            error_msg = str(e)
            # Check if it's a connection closed error (can appear in different formats)
            if "1000" in error_msg or "connection closed" in error_msg.lower() or "connection is closed" in error_msg.lower():
                logger.debug(f"🔌 Connection closed while sending audio (will reconnect)")
            else:
                logger.warning(f"⚠️  Failed to send audio to Gemini: {e}")
            # Don't clear session on error - let it reconnect automatically

    def _get_streamsdk_queue_stats(self) -> dict:
        """Get StreamSDK internal queue statistics."""
        if not self.sdk:
            return {}
        
        stats = {}
        try:
            # Check if queues exist and get their sizes
            if hasattr(self.sdk, 'audio2motion_queue'):
                stats['audio2motion_queue'] = {
                    'size': self.sdk.audio2motion_queue.qsize(),
                    'maxsize': self.sdk.audio2motion_queue.maxsize
                }
            if hasattr(self.sdk, 'motion_stitch_queue'):
                stats['motion_stitch_queue'] = {
                    'size': self.sdk.motion_stitch_queue.qsize(),
                    'maxsize': self.sdk.motion_stitch_queue.maxsize
                }
            if hasattr(self.sdk, 'warp_f3d_queue'):
                stats['warp_f3d_queue'] = {
                    'size': self.sdk.warp_f3d_queue.qsize(),
                    'maxsize': self.sdk.warp_f3d_queue.maxsize
                }
            if hasattr(self.sdk, 'decode_f3d_queue'):
                stats['decode_f3d_queue'] = {
                    'size': self.sdk.decode_f3d_queue.qsize(),
                    'maxsize': self.sdk.decode_f3d_queue.maxsize
                }
            if hasattr(self.sdk, 'putback_queue'):
                stats['putback_queue'] = {
                    'size': self.sdk.putback_queue.qsize(),
                    'maxsize': self.sdk.putback_queue.maxsize
                }
        except Exception as e:
            logger.debug(f"Could not get StreamSDK queue stats: {e}")
        
        return stats

    def _log_streamsdk_status(self):
        """Log StreamSDK status including queue states."""
        if not self.sdk:
            return
        
        logger.info("=" * 70)
        logger.info("📊 STREAMSDK STATUS")
        logger.info("=" * 70)
        logger.info(f"  Frames generated: {self._frames_generated}")
        logger.info(f"  Frames dropped: {self._frames_dropped}")
        logger.info(f"  Ditto chunks sent: {self._ditto_chunks_sent}")
        logger.info(f"  Online mode: {self.sdk.online_mode}")
        logger.info(f"  Streaming mode: {self.sdk.streaming_mode}")
        
        # Queue statistics
        queue_stats = self._get_streamsdk_queue_stats()
        if queue_stats:
            logger.info("  Queue states:")
            for queue_name, queue_info in queue_stats.items():
                usage_pct = (queue_info['size'] / queue_info['maxsize'] * 100) if queue_info['maxsize'] > 0 else 0
                logger.info(f"    {queue_name}: {queue_info['size']}/{queue_info['maxsize']} ({usage_pct:.1f}%)")
                if usage_pct > 80:
                    logger.warning(f"    ⚠️  {queue_name} is {usage_pct:.1f}% full - potential bottleneck!")
        
        # Check for worker exceptions
        if hasattr(self.sdk, 'worker_exception') and self.sdk.worker_exception:
            logger.error(f"  ❌ Worker exception: {self.sdk.worker_exception}")
        
        logger.info("=" * 70)

    def _log_profile_summary(self):
        """Log profiling statistics summary."""
        if not self._enable_profiling:
            return
        
        stats = self._get_profile_stats()
        if not stats:
            return
        
        logger.info("=" * 70)
        logger.info("📊 PROFILING SUMMARY")
        logger.info("=" * 70)
        for stage, data in stats.items():
            logger.info(f"  {stage}:")
            logger.info(f"    Count: {data['count']}")
            logger.info(f"    Avg: {data['avg_ms']:.2f}ms")
            logger.info(f"    Min: {data['min_ms']:.2f}ms")
            logger.info(f"    Max: {data['max_ms']:.2f}ms")
            logger.info(f"    Total: {data['total_ms']:.2f}ms")
        logger.info("=" * 70)

    def close(self):
        """Cleanup resources."""
        # Log final StreamSDK status
        self._log_streamsdk_status()
        
        # Log final profiling summary
        self._log_profile_summary()
        
        if self.sdk:
            try:
                logger.info("🔧 Closing StreamSDK...")
                self.sdk.close()
                logger.info("✅ StreamSDK closed")
            except Exception as e:
                logger.error(f"❌ Error closing StreamSDK: {e}")
                import traceback
                logger.error(traceback.format_exc())


async def entrypoint(ctx: JobContext):
    """
    LiveKit + Gemini agent entrypoint.

    Environment variables:
    - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    - Authentication (choose one):
      * GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON) - REQUIRED for Vertex AI
      * VERTEX_PROJECT_ID (GCP project ID) - REQUIRED for Vertex AI
      * VERTEX_LOCATION (optional, default: us-central1)
      * OR GEMINI_API_KEY (for API key auth)
    - DITTO_CFG_PKL, DITTO_DATA_ROOT, DITTO_SOURCE
    - DITTO_MAX_SIZE, DITTO_EMO
    - GEMINI_MODEL (optional, default: gemini-live-2.5-flash-preview-native-audio-09-2025)
    - GEMINI_VOICE (optional, default: Puck)
    - GEMINI_INSTRUCTION (optional system instruction)
    
    Note: Vertex AI is prioritized if GOOGLE_APPLICATION_CREDENTIALS is set,
    even if GEMINI_API_KEY is also present.
    """
    logger.info(f"🚀 Gemini agent starting for room: {ctx.room.name}")

    # Configuration
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    vertex_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    vertex_project = os.getenv("VERTEX_PROJECT_ID")

    # Require either API key OR Vertex AI credentials
    if not gemini_api_key and not vertex_creds:
        raise ValueError(
            "Either GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS environment variable required.\n"
            "For Vertex AI, also set VERTEX_PROJECT_ID."
        )
    
    # Log which auth method will be used
    if vertex_creds:
        logger.info("🔐 Vertex AI authentication detected")
        if not vertex_project:
            logger.warning("⚠️  VERTEX_PROJECT_ID not set - authentication may fail")
    elif gemini_api_key:
        logger.info("🔑 API key authentication detected")

    cfg_pkl = os.getenv("DITTO_CFG_PKL", "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    data_root = os.getenv("DITTO_DATA_ROOT", "checkpoints/ditto_trt_custom2/")
    source_path = os.getenv("DITTO_SOURCE", "avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg")
    max_size = int(os.getenv("DITTO_MAX_SIZE", "1920"))
    emo = int(os.getenv("DITTO_EMO", "4"))

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025")
    gemini_voice = os.getenv("GEMINI_VOICE", "Puck")
    system_instruction = os.getenv("GEMINI_INSTRUCTION", "You are a helpful AI assistant. Keep responses concise and natural.")

    logger.info(f"📋 Configuration:")
    logger.info(f"   Ditto: {source_path}")
    logger.info(f"   Gemini: {gemini_model} (voice: {gemini_voice})")

    # Create agent
    agent = GeminiDittoAgent(
        cfg_pkl=cfg_pkl,
        data_root=data_root,
        source_path=source_path,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        voice_name=gemini_voice,
        system_instruction=system_instruction,
        max_size=max_size,
        emo=emo
    )

    # Initialize
    await agent.initialize()

    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.room = ctx.room

    # Create video source
    agent.video_source = rtc.VideoSource(1280, 720)
    video_track = rtc.LocalVideoTrack.create_video_track("ditto_avatar", agent.video_source)
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(video_track, video_options)

    logger.info("✅ Video track published")

    # Create audio source (for avatar's speech)
    agent.audio_source = rtc.AudioSource(48000, 1)  # 48kHz, mono
    audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_voice", agent.audio_source)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.room.local_participant.publish_track(audio_track, audio_options)

    logger.info("✅ Audio track published (you'll hear the avatar speak!)")

    # Start background tasks
    gemini_task = asyncio.create_task(agent.start_gemini_session())
    silent_audio_task = asyncio.create_task(agent.run_silent_audio_generator())

    logger.info("✅ Background tasks started (Gemini session + silent audio generator)")

    # Subscribe to user audio
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"📡 Subscribed to {track.kind} from {participant.identity}")

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("🎤 Processing user audio")

            audio_stream = rtc.AudioStream(track)

            async def process_audio_stream():
                async for audio_frame_event in audio_stream:
                    # Send to Gemini for conversation
                    await agent.send_user_audio_to_gemini(audio_frame_event.frame)

            asyncio.create_task(process_audio_stream())

    logger.info("✅ Agent ready - speak to start conversation!")
    logger.info("💡 Conversation features:")
    logger.info("   - Avatar always visible with smooth animation at 25 FPS")
    logger.info("   - Gemini handles VAD and turn detection automatically")
    logger.info("   - Speak naturally - Gemini detects when you start/stop")
    logger.info("   - AI responds automatically when you finish speaking")
    logger.info("   - Supports interruptions (you can speak while AI is speaking)")
    logger.info("🎬 Video settings:")
    logger.info(f"   - Target FPS: 25 (steady for both IDLE and SPEAKING)")
    logger.info(f"   - Max FPS: 40 (before frame dropping)")
    logger.info(f"   - Audio chunk: 6480 samples @ 16kHz")
    logger.info(f"   - Chunksize: (3, 5, 2) → 5 frames per chunk")
    logger.info(f"   - Feed rate: 5 Hz → steady 25 FPS")

    # Keep alive and wait for both tasks
    try:
        await asyncio.gather(gemini_task, silent_audio_task)
    except asyncio.CancelledError:
        pass
    finally:
        # Cancel tasks
        gemini_task.cancel()
        silent_audio_task.cancel()
        try:
            await gemini_task
        except asyncio.CancelledError:
            pass
        try:
            await silent_audio_task
        except asyncio.CancelledError:
            pass
        agent.close()


async def request_fnc(ctx: JobContext):
    """Request handler."""
    logger.info(f"📩 Job request for room: {ctx.room.name}")
    await ctx.accept()


if __name__ == "__main__":
    """
    Start the LiveKit + Gemini agent.

    Required environment variables:
    - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    - Authentication (choose one):
      * GOOGLE_APPLICATION_CREDENTIALS + VERTEX_PROJECT_ID (for Vertex AI)
      * OR GEMINI_API_KEY (for API key auth)

    Optional:
    - DITTO_* variables (same as livekit_ditto_agent.py)
    - GEMINI_MODEL, GEMINI_VOICE, GEMINI_INSTRUCTION
    - VERTEX_LOCATION (default: us-central1)
    """
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
        )
    )
