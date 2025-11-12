"""
LiveKit + Vertex AI (STT->LLM->TTS) + Ditto Conversational Avatar Agent

This agent uses a cascaded approach:
1. STT: Google Speech-to-Text (Vertex AI)
2. LLM: Gemini 2.0 Flash (Vertex AI)
3. TTS: Google Text-to-Speech (Vertex AI)
4. Video: Ditto avatar generation

Usage:
    export LIVEKIT_URL=ws://localhost:7880
    export LIVEKIT_API_KEY=devkey
    export LIVEKIT_API_SECRET=devsecret
    export GOOGLE_APPLICATION_CREDENTIALS=gnani-video-ai-c3b9b902d4d8.json
    export VERTEX_PROJECT_ID=gnani-video-ai
    export VERTEX_LOCATION=us-central1

    python livekit_vertex_cascade_agent.py dev
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# LiveKit
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)

# Google Cloud
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as texttospeech
from google import genai

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stream_pipeline_online import StreamSDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log authentication config
def _log_auth_config():
    """Log which authentication method is configured."""
    vertex_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    vertex_project = os.getenv("VERTEX_PROJECT_ID")

    if vertex_creds:
        logger.info("=" * 60)
        logger.info("🔐 Vertex AI Authentication Configured")
        logger.info(f"   Credentials: {vertex_creds}")
        logger.info(f"   Project ID:  {vertex_project or 'NOT SET ⚠️'}")
        logger.info(f"   Location:    {os.getenv('VERTEX_LOCATION', 'us-central1')}")
        logger.info("=" * 60)
    else:
        logger.warning("⚠️  No authentication configured!")

_log_auth_config()


class VertexCascadeAgent:
    """
    Conversational avatar agent using Vertex AI cascade (STT->LLM->TTS) + Ditto.

    Flow:
    1. User audio → Google Speech-to-Text → Text
    2. Text → Gemini LLM → Response text
    3. Response text → Google Text-to-Speech → Audio
    4. TTS audio → Ditto model → Video frames
    5. Video + Audio → LiveKit → Browser
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        project_id: str,
        location: str = "us-central1",
        gemini_model: str = "gemini-2.0-flash-exp",
        tts_voice: str = "en-US-Neural2-F",
        system_instruction: str = None,
        **ditto_kwargs
    ):
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.source_path = source_path
        self.project_id = project_id
        self.location = location
        self.gemini_model = gemini_model
        self.tts_voice = tts_voice
        self.system_instruction = system_instruction or "You are a helpful AI assistant. Keep responses concise and natural."
        self.ditto_kwargs = ditto_kwargs

        # Components
        self.sdk: Optional[StreamSDK] = None
        self.stt_client: Optional[speech.SpeechClient] = None
        self.tts_client: Optional[texttospeech.TextToSpeechClient] = None
        self.gemini_client = None

        # LiveKit
        self.room: Optional[rtc.Room] = None
        self.video_source: Optional[rtc.VideoSource] = None
        self.audio_source: Optional[rtc.AudioSource] = None

        # Audio buffers
        self.stt_audio_buffer = b""
        self.tts_audio_buffer = np.array([], dtype=np.float32)
        self.model_chunk_size = 6400  # 400ms @ 16kHz

        # Conversation state
        self.conversation_history = []
        self.is_processing = False

        # Frame timing
        self.target_fps = 25
        self.last_frame_time = None
        self.frame_interval = 1.0 / self.target_fps

        # Statistics & latency tracking
        self._frames_generated = 0
        self._frames_dropped = 0
        self._stt_calls = 0
        self._llm_calls = 0
        self._tts_calls = 0

        self._stt_latency = []
        self._llm_latency = []
        self._tts_latency = []
        self._total_latency = []

    async def initialize(self):
        """Initialize all components."""
        logger.info("🎭 Initializing Ditto SDK...")
        await asyncio.to_thread(self._init_sdk_sync)
        logger.info("✅ Ditto SDK initialized")

        logger.info("🎤 Initializing Google Speech-to-Text...")
        self.stt_client = speech.SpeechClient()
        logger.info("✅ STT client initialized")

        logger.info("🔊 Initializing Google Text-to-Speech...")
        self.tts_client = texttospeech.TextToSpeechClient()
        logger.info("✅ TTS client initialized")

        logger.info("🤖 Initializing Gemini LLM...")
        self.gemini_client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        logger.info(f"✅ Gemini client initialized (model: {self.gemini_model})")

    def _init_sdk_sync(self):
        """Synchronous SDK initialization."""
        self.sdk = StreamSDK(self.cfg_pkl, self.data_root, **self.ditto_kwargs)

        setup_kwargs = {
            "online_mode": True,
            "N_d": -1,
        }
        setup_kwargs.update(self.ditto_kwargs)

        self.sdk.setup(
            self.source_path,
            output_path=None,
            frame_callback=self._on_frame_generated,
            **setup_kwargs
        )

    def _on_frame_generated(self, frame_rgb: np.ndarray, frame_idx: int, timestamp: float):
        """Callback when Ditto generates a frame (from worker thread)."""
        self._frames_generated += 1

        if self._frames_generated == 1:
            logger.info(f"🎬 First frame generated: {frame_rgb.shape}")
            self.last_frame_time = time.time()

        # Frame pacing
        current_time = time.time()
        if self.last_frame_time:
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_interval * 0.8:
                self._frames_dropped += 1
                return

        self.last_frame_time = current_time

        if self.video_source:
            try:
                video_frame = rtc.VideoFrame(
                    width=frame_rgb.shape[1],
                    height=frame_rgb.shape[0],
                    type=rtc.VideoBufferType.RGBA,
                    data=self._rgb_to_rgba(frame_rgb)
                )
                self.video_source.capture_frame(video_frame)

                if self._frames_generated % 100 == 0:
                    logger.info(f"📊 Frames: {self._frames_generated} sent, {self._frames_dropped} dropped")

            except Exception as e:
                logger.error(f"Error sending frame: {e}")

    def _rgb_to_rgba(self, rgb: np.ndarray) -> bytes:
        """Convert RGB to RGBA."""
        h, w, _ = rgb.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        return rgba.tobytes()

    async def process_audio_chunk(self, audio_data: bytes):
        """
        Process an audio chunk through the cascade:
        STT -> LLM -> TTS -> Ditto
        """
        if self.is_processing:
            logger.debug("⏳ Already processing, skipping chunk")
            return

        self.is_processing = True
        start_time = time.time()

        try:
            # Step 1: STT (Speech-to-Text)
            stt_start = time.time()
            text = await self._stt_transcribe(audio_data)
            stt_time = time.time() - stt_start

            if not text or len(text.strip()) < 3:
                logger.debug("🔇 No significant speech detected")
                return

            logger.info(f"👤 User said: {text}")
            self._stt_calls += 1
            self._stt_latency.append(stt_time)

            # Step 2: LLM (Gemini)
            llm_start = time.time()
            response_text = await self._llm_generate(text)
            llm_time = time.time() - llm_start

            logger.info(f"🤖 Assistant: {response_text}")
            self._llm_calls += 1
            self._llm_latency.append(llm_time)

            # Step 3: TTS (Text-to-Speech)
            tts_start = time.time()
            audio_bytes = await self._tts_synthesize(response_text)
            tts_time = time.time() - tts_start

            self._tts_calls += 1
            self._tts_latency.append(tts_time)

            # Step 4: Send TTS audio to browser & Ditto
            await self._process_tts_audio(audio_bytes)

            # Log latency
            total_time = time.time() - start_time
            self._total_latency.append(total_time)

            logger.info(f"⏱️  Latency breakdown:")
            logger.info(f"   STT: {stt_time*1000:.0f}ms")
            logger.info(f"   LLM: {llm_time*1000:.0f}ms")
            logger.info(f"   TTS: {tts_time*1000:.0f}ms")
            logger.info(f"   TOTAL: {total_time*1000:.0f}ms")

        except Exception as e:
            logger.error(f"❌ Error in cascade: {e}", exc_info=True)
        finally:
            self.is_processing = False

    async def _stt_transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using Google Speech-to-Text."""
        def _sync_transcribe():
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )

            audio = speech.RecognitionAudio(content=audio_data)
            response = self.stt_client.recognize(config=config, audio=audio)

            if response.results:
                return response.results[0].alternatives[0].transcript
            return ""

        return await asyncio.to_thread(_sync_transcribe)

    async def _llm_generate(self, user_text: str) -> str:
        """Generate response using Gemini."""
        def _sync_generate():
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_text})

            # Build prompt with history
            prompt = f"{self.system_instruction}\n\n"
            for msg in self.conversation_history[-10:]:  # Last 10 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant: "

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )

            response_text = response.text.strip() if response.text else "I'm sorry, I couldn't generate a response."

            # Add to history
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text

        return await asyncio.to_thread(_sync_generate)

    async def _tts_synthesize(self, text: str) -> bytes:
        """Synthesize speech using Google Text-to-Speech."""
        def _sync_synthesize():
            synthesis_input = texttospeech.SynthesisInput(text=text)

            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=self.tts_voice,
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            )

            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            return response.audio_content

        return await asyncio.to_thread(_sync_synthesize)

    async def _process_tts_audio(self, audio_bytes: bytes):
        """Process TTS audio: send to browser and feed to Ditto."""
        # Convert to numpy array (24kHz PCM16)
        audio_data_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_data = audio_data_int16.astype(np.float32) / 32768.0

        # Send to browser (48kHz)
        if self.audio_source:
            import scipy.signal
            audio_48khz = scipy.signal.resample_poly(audio_data, up=2, down=1)
            audio_48khz_int16 = (audio_48khz * 32768).astype(np.int16)

            samples_per_chunk = 48000 // 50  # 960 samples = 20ms @ 48kHz
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

        # Feed to Ditto (16kHz)
        import scipy.signal
        audio_16khz = scipy.signal.resample_poly(audio_data, up=2, down=3)

        # Add to buffer and process chunks
        self.tts_audio_buffer = np.concatenate([self.tts_audio_buffer, audio_16khz])

        while len(self.tts_audio_buffer) >= self.model_chunk_size:
            chunk = self.tts_audio_buffer[:self.model_chunk_size]
            self.tts_audio_buffer = self.tts_audio_buffer[self.model_chunk_size:]

            await asyncio.to_thread(
                self.sdk.run_chunk,
                chunk,
                (3, 5, 2)
            )

    def print_stats(self):
        """Print latency statistics."""
        logger.info("=" * 60)
        logger.info("📊 Performance Statistics")
        logger.info("=" * 60)

        if self._stt_latency:
            avg_stt = np.mean(self._stt_latency) * 1000
            logger.info(f"STT: {self._stt_calls} calls, avg {avg_stt:.0f}ms")

        if self._llm_latency:
            avg_llm = np.mean(self._llm_latency) * 1000
            logger.info(f"LLM: {self._llm_calls} calls, avg {avg_llm:.0f}ms")

        if self._tts_latency:
            avg_tts = np.mean(self._tts_latency) * 1000
            logger.info(f"TTS: {self._tts_calls} calls, avg {avg_tts:.0f}ms")

        if self._total_latency:
            avg_total = np.mean(self._total_latency) * 1000
            min_total = np.min(self._total_latency) * 1000
            max_total = np.max(self._total_latency) * 1000
            logger.info(f"TOTAL: avg {avg_total:.0f}ms, min {min_total:.0f}ms, max {max_total:.0f}ms")

        logger.info("=" * 60)

    def close(self):
        """Cleanup resources."""
        self.print_stats()

        if self.sdk:
            try:
                self.sdk.close()
                logger.info("✅ Ditto SDK closed")
            except Exception as e:
                logger.error(f"Error closing SDK: {e}")


async def entrypoint(ctx: JobContext):
    """
    LiveKit + Vertex AI Cascade agent entrypoint.

    Environment variables:
    - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    - GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
    - VERTEX_PROJECT_ID (GCP project ID)
    - VERTEX_LOCATION (optional, default: us-central1)
    - DITTO_CFG_PKL, DITTO_DATA_ROOT, DITTO_SOURCE
    - DITTO_MAX_SIZE, DITTO_EMO
    - GEMINI_MODEL (optional, default: gemini-2.0-flash-exp)
    - TTS_VOICE (optional, default: en-US-Neural2-F)
    - SYSTEM_INSTRUCTION (optional)
    """
    logger.info(f"🚀 Vertex Cascade agent starting for room: {ctx.room.name}")

    # Configuration
    project_id = os.getenv("VERTEX_PROJECT_ID")
    if not project_id:
        raise ValueError("VERTEX_PROJECT_ID environment variable required")

    location = os.getenv("VERTEX_LOCATION", "us-central1")
    cfg_pkl = os.getenv("DITTO_CFG_PKL", "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
    data_root = os.getenv("DITTO_DATA_ROOT", "checkpoints/ditto_trt_custom2/")
    source_path = os.getenv("DITTO_SOURCE", "avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg")
    max_size = int(os.getenv("DITTO_MAX_SIZE", "1920"))
    emo = int(os.getenv("DITTO_EMO", "4"))

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    tts_voice = os.getenv("TTS_VOICE", "en-US-Neural2-F")
    system_instruction = os.getenv("SYSTEM_INSTRUCTION", "You are a helpful AI assistant. Keep responses concise and natural.")

    logger.info(f"📋 Configuration:")
    logger.info(f"   Ditto: {source_path}")
    logger.info(f"   Gemini: {gemini_model}")
    logger.info(f"   TTS Voice: {tts_voice}")

    # Create agent
    agent = VertexCascadeAgent(
        cfg_pkl=cfg_pkl,
        data_root=data_root,
        source_path=source_path,
        project_id=project_id,
        location=location,
        gemini_model=gemini_model,
        tts_voice=tts_voice,
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

    logger.info("✅ Audio track published")

    # Buffer for accumulating audio before processing
    audio_accumulator = []
    min_audio_duration = 2.0  # seconds
    sample_rate = 16000

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
                    frame = audio_frame_event.frame

                    # Convert to 16kHz mono
                    audio_data = np.frombuffer(frame.data, dtype=np.int16)
                    if frame.num_channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                    # Resample to 16kHz if needed
                    if frame.sample_rate != 16000:
                        import scipy.signal
                        audio_float = audio_data.astype(np.float32) / 32768.0
                        audio_float = scipy.signal.resample_poly(
                            audio_float,
                            up=16000,
                            down=frame.sample_rate
                        )
                        audio_data = (audio_float * 32768).astype(np.int16)

                    audio_accumulator.append(audio_data)

                    # Process when we have enough audio
                    total_samples = sum(len(chunk) for chunk in audio_accumulator)
                    duration = total_samples / sample_rate

                    if duration >= min_audio_duration:
                        # Concatenate and process
                        full_audio = np.concatenate(audio_accumulator)
                        audio_bytes = full_audio.tobytes()
                        audio_accumulator.clear()

                        # Process through cascade
                        asyncio.create_task(agent.process_audio_chunk(audio_bytes))

            asyncio.create_task(process_audio_stream())

    logger.info("✅ Agent ready - speak to start conversation!")

    # Keep alive
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        agent.close()


async def request_fnc(ctx: JobContext):
    """Request handler."""
    logger.info(f"📩 Job request for room: {ctx.room.name}")
    await ctx.accept()


if __name__ == "__main__":
    """
    Start the LiveKit + Vertex AI Cascade agent.

    Required environment variables:
    - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    - GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
    - VERTEX_PROJECT_ID (GCP project ID)

    Optional:
    - VERTEX_LOCATION (default: us-central1)
    - DITTO_* variables
    - GEMINI_MODEL, TTS_VOICE, SYSTEM_INSTRUCTION
    """
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
        )
    )
