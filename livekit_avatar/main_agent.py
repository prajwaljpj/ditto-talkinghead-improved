import os
import asyncio
import logging
import time
import numpy as np
from livekit import agents, rtc
from livekit.agents import AgentStateChangedEvent
from livekit.plugins import google

# Import custom avatar worker
from custom_avatar_worker import CustomAvatarWorker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("websockets").setLevel(logging.ERROR)

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
DATA_ROOT = os.environ.get("DATA_ROOT", "./checkpoints/ditto_trt_Ampere_Plus")
CFG_PKL = os.environ.get("CFG_PKL", "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
SOURCE_PATH = os.environ.get("SOURCE_PATH", "./assets/source_image.png")

AVATAR_WIDTH = int(os.environ.get("AVATAR_WIDTH", "1280"))
AVATAR_HEIGHT = int(os.environ.get("AVATAR_HEIGHT", "720"))


async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the LiveKit Agent.

    This agent:
    1. Connects to a room
    2. Listens to user audio via AgentSession
    3. Generates responses using Google's Gemini model
    4. Captures TTS audio from the model and feeds it to the avatar
    5. Publishes avatar video back to the room
    """
    logger.info(f"Starting avatar agent in room: {ctx.room.name}")

    # 1. Connect to the room first
    await ctx.connect()
    logger.info(f"✅ Connected to room: {ctx.room.name}")

    # 2. Create synchronized audio and video sources for the avatar
    # These will be separate from the AgentSession's audio (which handles conversation)
    video_source = rtc.VideoSource(AVATAR_WIDTH, AVATAR_HEIGHT)
    audio_source = rtc.AudioSource(
        sample_rate=16000,  # Match Ditto's expected sample rate
        num_channels=1,  # Mono
        queue_size_ms=1000,
    )

    # 3. Create AVSynchronizer to keep audio and video in sync
    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=50,  # Match our avatar FPS
        video_queue_size_ms=2000,  # Buffer up to 2 seconds (increased for slower GPUs)
    )
    logger.info("✅ Created AVSynchronizer for audio/video sync")

    # 4. Create and publish tracks
    video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", video_source)
    audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", audio_source)

    video_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_encoding=rtc.VideoEncoding(
            max_framerate=50,
            max_bitrate=5_000_000,  # 5 Mbps for HD quality
        ),
    )
    audio_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
    )

    await ctx.room.local_participant.publish_track(video_track, video_options)
    await ctx.room.local_participant.publish_track(audio_track, audio_options)
    logger.info(
        f"✅ Published synchronized audio+video tracks: {AVATAR_WIDTH}x{AVATAR_HEIGHT}"
    )

    # 5. Initialize the custom avatar worker
    # This worker will generate video frames synchronized with audio
    avatar_worker = CustomAvatarWorker(
        data_root=DATA_ROOT,
        cfg_pkl=CFG_PKL,
        source_path=SOURCE_PATH,
        frame_width=AVATAR_WIDTH,
        frame_height=AVATAR_HEIGHT,
        av_sync=av_sync,  # Pass AVSynchronizer instead of raw video_source
    )

    # 6. Start the avatar worker (begins with idle/silent mode)
    avatar_worker.start()
    logger.info("✅ Avatar worker started (idle mode)")

    # 7. Create the LLM model
    llm_model = google.beta.realtime.RealtimeModel(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        model=os.environ.get(
            "GEMINI_MODEL", "gemini-live-2.5-flash-preview-native-audio-09-2025"
        ),
        voice="Charon",
    )

    # 8. Create the voice agent with instructions and LLM
    voice_agent = agents.Agent(
        instructions=(
            "You are a helpful AI assistant with an animated avatar. "
            "Keep your responses conversational and concise. "
            "Speak naturally as if you're having a face-to-face conversation."
        ),
        llm=llm_model,
    )

    # 9. Initialize the agent session
    main_agent_session = agents.AgentSession()

    # 10. Set up event handlers to track conversation state

    ## new code from docs ##
    # The agent state events are AgentState (enum) : ["initializing", "listening", "thinking", "speaking"]
    @main_agent_session.on("agent_state_changed")
    def on_agent_state_change(event: AgentStateChangedEvent):
        logger.info(
            f"Change in Agent State from {event.old_state} --> {event.new_state}"
        )
        avatar_worker.set_state(event.new_state)

    ## Old code from claude ##
    # @main_agent_session.on("user_turn_started")
    # def on_user_started_speaking(data):
    #     logger.info("🎤 User started speaking (Avatar -> Listening mode)")
    #     avatar_worker.set_state("listening")

    # @main_agent_session.on("user_turn_completed")
    # def on_user_stopped_speaking(data):
    #     logger.info("🤔 User finished speaking (Avatar -> Thinking mode)")
    #     avatar_worker.set_state("thinking")

    # @main_agent_session.on("agent_started_speaking")
    # def on_agent_started_speaking(data):
    #     logger.info("🗣️ Agent started speaking (Avatar -> Speaking mode)")
    #     avatar_worker.set_state("speaking")

    # @main_agent_session.on("agent_stopped_speaking")
    # def on_agent_stopped_speaking(data):
    #     logger.info("✅ Agent finished speaking (Avatar -> Idle mode)")
    #     avatar_worker.set_state("idle")

    # 11. Set up synchronized audio capture from the AgentSession's TTS output
    # We capture the audio and push it through AVSynchronizer for proper sync
    async def capture_and_sync_audio():
        """
        Captures TTS audio and feeds it to:
        1. Avatar worker (for video generation)
        2. AVSynchronizer (for synchronized playback)

        Note: We unpublish the AgentSession's TTS audio track to prevent
        duplicate audio. Clients will only hear the synchronized avatar_audio.
        """
        # Wait for the agent to publish its audio track
        await asyncio.sleep(1.0)

        # Find the agent's audio track (AgentSession's TTS output)
        for pub in ctx.room.local_participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track:
                # Skip our own avatar_audio track
                if pub.name == "avatar_audio":
                    continue

                logger.info(f"✅ Found agent TTS audio track: {pub.name}")

                # Unpublish this track to prevent duplicate audio
                # Clients will only hear the synchronized avatar_audio track
                logger.info(f"🔇 Unpublishing AgentSession's audio track to avoid duplication")
                await ctx.room.local_participant.unpublish_track(pub.sid)

                # Continue capturing from the track (still works after unpublishing)
                audio_stream = rtc.AudioStream(pub.track)

                # Process audio frames
                async for frame_event in audio_stream:
                    try:
                        audio_frame = frame_event.frame

                        # Feed to avatar worker for video generation
                        # Note: Avatar worker now handles pushing both audio and video
                        # to AVSynchronizer to ensure perfect sync
                        await avatar_worker.feed_audio(audio_frame)

                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                break

    # 12. Start the AgentSession (connects VAD, STT, and TTS)
    await main_agent_session.start(voice_agent, room=ctx.room)
    logger.info("✅ Agent session started")

    # 13. Start capturing and synchronizing TTS audio in the background
    audio_sync_task = asyncio.create_task(capture_and_sync_audio())

    # 14. The agent will automatically handle the conversation flow
    logger.info("✅ Agent is ready and listening for user input")

    # 15. Keep the agent running until the room disconnects
    try:
        # Wait for the room to disconnect
        async def wait_until_disconnected():
            while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                await asyncio.sleep(1)

        await wait_until_disconnected()
        logger.info("Room disconnected, shutting down...")
    except Exception as e:
        logger.error(f"Agent session error: {e}")
    finally:
        # 16. Cleanup
        logger.info("Shutting down agent...")

        # Cancel audio sync task
        if audio_sync_task and not audio_sync_task.done():
            audio_sync_task.cancel()
            try:
                await audio_sync_task
            except asyncio.CancelledError:
                pass

        # Close avatar worker
        await avatar_worker.close()

        # Wait for AVSynchronizer to finish playout
        try:
            await av_sync.wait_for_playout()
        except Exception as e:
            logger.warning(f"Error waiting for playout: {e}")

        # Close AVSynchronizer
        await av_sync.aclose()

        logger.info("✅ Agent shutdown complete")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
