"""
Ditto Talking Head with Gemini Live API WebRTC Server

This server creates real-time conversational avatar using:
- Gemini Live API for ASR + LLM + TTS
- Ditto pipeline for talking head video generation
- Full-duplex conversation with emotion-aware expressions

Usage:
    export GEMINI_API_KEY="your-api-key"
    python gemini_server.py --cfg_pkl <path> --data_root <path> --source <avatar_image>
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Import our custom processors
from webrtc.processors.ditto_processor import DittoAvatarProcessor
from webrtc.processors.gemini_processor import GeminiLiveProcessor
from webrtc.processors.idle_animator import DittoIdleAnimationProcessor
from webrtc.conversation_manager import EmotionType


class GeminiDittoServer:
    """
    WebRTC server for Gemini + Ditto conversational avatar.

    Pipeline:
        User Audio → GeminiProcessor → Gemini Audio → DittoProcessor → Video
                                         ↓
                                    Emotion Detection
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        gemini_api_key: str = None,
        gemini_model: str = "models/gemini-2.0-flash-exp",
        voice_name: str = "Puck",
        enable_interruptions: bool = True,
        host: str = "0.0.0.0",
        port: int = 8080,
        **kwargs
    ):
        """
        Initialize Gemini + Ditto server.

        Args:
            cfg_pkl: Path to Ditto configuration pickle
            data_root: Path to model data root
            source_path: Path to avatar source image/video
            gemini_api_key: Gemini API key (or use GEMINI_API_KEY env var)
            gemini_model: Gemini model name
            voice_name: Gemini voice (Puck, Charon, Kore, Fenrir, Aoede)
            enable_interruptions: Allow user to interrupt avatar
            host: Server host address
            port: Server port
            **kwargs: Additional Ditto configuration
        """
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.source_path = source_path
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model
        self.voice_name = voice_name
        self.enable_interruptions = enable_interruptions
        self.host = host
        self.port = port
        self.ditto_kwargs = kwargs

        # Current emotion (shared between Gemini and Ditto)
        self.current_emotion = EmotionType.NEUTRAL

        # Processors
        self.gemini_processor = None
        self.ditto_processor = None

        self.runner = None

    def on_emotion_change(self, emotion: EmotionType):
        """
        Callback when Gemini detects emotion change.

        This updates the Ditto processor to use appropriate facial expressions.
        """
        self.current_emotion = emotion
        print(f"[GeminiDittoServer] Emotion changed to: {emotion.value}")

        # Update Ditto emotion parameter
        # Note: This requires modifying DittoProcessor to support dynamic emotion updates
        if self.ditto_processor and hasattr(self.ditto_processor, 'update_emotion'):
            self.ditto_processor.update_emotion(emotion.value)

    async def create_pipeline(self):
        """
        Create the Pipecat pipeline with Gemini + Ditto processors.

        Pipeline structure:
            User Audio → Gemini → Gemini Audio → Ditto → Video
                            ↓
                        Emotion Detection
        """

        # Create Gemini Live processor
        self.gemini_processor = GeminiLiveProcessor(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            voice_name=self.voice_name,
            enable_interruptions=self.enable_interruptions,
            on_emotion_change=self.on_emotion_change,
        )

        # Create Ditto avatar processor
        # Start with the current emotion
        ditto_kwargs = self.ditto_kwargs.copy()
        if 'emo' not in ditto_kwargs:
            # Map emotion to Ditto emotion code (0-7)
            # Default to neutral (4)
            emotion_map = {
                EmotionType.NEUTRAL: 4,
                EmotionType.HAPPY: 0,
                EmotionType.SAD: 2,
                EmotionType.ANGRY: 1,
                EmotionType.SURPRISED: 5,
            }
            ditto_kwargs['emo'] = emotion_map.get(self.current_emotion, 4)

        self.ditto_processor = DittoAvatarProcessor(
            cfg_pkl=self.cfg_pkl,
            data_root=self.data_root,
            source_path=self.source_path,
            **ditto_kwargs
        )

        # Create idle animation processor
        idle_animator = DittoIdleAnimationProcessor(
            idle_threshold=0.01,
            idle_timeout=0.5,
            fps=25
        )

        # Create pipeline
        # Audio flows: User → Gemini → Ditto → Video
        pipeline = Pipeline([
            self.gemini_processor,
            self.ditto_processor,
            idle_animator,
        ])

        return pipeline

    async def run(self):
        """
        Run the server (without WebRTC transport for now).

        This will use signaling_server.py for actual WebRTC connections.
        This method is for testing the pipeline integration.
        """
        print("=" * 70)
        print("Gemini + Ditto Conversational Avatar Server")
        print("=" * 70)
        print(f"Gemini Model: {self.gemini_model}")
        print(f"Voice: {self.voice_name}")
        print(f"Ditto Config: {self.cfg_pkl}")
        print(f"Avatar Source: {self.source_path}")
        print(f"Interruptions: {'Enabled' if self.enable_interruptions else 'Disabled'}")
        print("=" * 70)

        if not self.gemini_api_key:
            print("\nError: GEMINI_API_KEY not set!")
            print("Set it with: export GEMINI_API_KEY='your-api-key'")
            sys.exit(1)

        print("\nInitializing pipeline...")
        pipeline = await self.create_pipeline()

        # Create task params
        params = PipelineParams(
            allow_interruptions=self.enable_interruptions,
            enable_metrics=True,
            enable_usage_metrics=False,
        )

        # Create task
        task = PipelineTask(pipeline, params=params)

        # Create runner
        self.runner = PipelineRunner()

        print("\nPipeline ready!")
        print("\nNote: This is a test runner for the Pipecat pipeline.")
        print("For full WebRTC functionality, use the signaling_server.py")
        print("which integrates this pipeline with WebRTC connections.")
        print("\nPress Ctrl+C to stop...")
        print()

        # Run
        await self.runner.run(task)

    async def shutdown(self):
        """Shutdown the server gracefully."""
        if self.runner:
            await self.runner.stop()

        # Get conversation summary
        if self.gemini_processor:
            summary = await self.gemini_processor.get_conversation_summary()
            print("\n" + "=" * 70)
            print("Conversation Summary:")
            print("=" * 70)
            print(f"Session ID: {summary.get('session_id', 'N/A')}")
            print(f"Total Turns: {summary.get('turn_count', 0)}")
            print(f"Final Emotion: {summary.get('current_emotion', 'N/A')}")
            print(f"Duration: {summary.get('last_activity', 0):.1f}s")
            print("\nRecent turns:")
            for turn in summary.get('recent_turns', []):
                print(f"  [{turn['role']}] {turn['text']} ({turn['emotion']})")
            print("=" * 70)

        print("\nServer shutdown complete.")


async def main():
    parser = argparse.ArgumentParser(
        description="Gemini + Ditto Conversational Avatar Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires GEMINI_API_KEY environment variable):
  export GEMINI_API_KEY="your-api-key"
  python gemini_server.py \\
    --cfg_pkl outputs/cfg_f_model.pkl \\
    --data_root ./ \\
    --source examples/avatar.jpg

  # With custom voice and emotion settings:
  python gemini_server.py \\
    --cfg_pkl outputs/cfg_f_model.pkl \\
    --data_root ./ \\
    --source examples/avatar.jpg \\
    --voice Aoede \\
    --emo 0

  # Disable interruptions for turn-based conversation:
  python gemini_server.py \\
    --cfg_pkl outputs/cfg_f_model.pkl \\
    --data_root ./ \\
    --source examples/avatar.jpg \\
    --no-interruptions
        """
    )

    # Required arguments
    parser.add_argument("--cfg_pkl", required=True, help="Path to config pickle file")
    parser.add_argument("--data_root", required=True, help="Path to model data root directory")
    parser.add_argument("--source", required=True, help="Path to avatar source image or video")

    # Gemini configuration
    parser.add_argument("--api_key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="models/gemini-2.0-flash-exp", help="Gemini model name")
    parser.add_argument("--voice", default="Puck",
                        choices=["Puck", "Charon", "Kore", "Fenrir", "Aoede"],
                        help="Gemini voice for TTS")
    parser.add_argument("--no-interruptions", action="store_true",
                        help="Disable interruptions (turn-based conversation)")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    # Ditto configuration
    parser.add_argument("--max_size", type=int, default=1920, help="Max image dimension")
    parser.add_argument("--crop_scale", type=float, default=2.3, help="Face crop scale")
    parser.add_argument("--emo", type=int, default=4,
                        help="Initial emotion (0=happy, 1=angry, 2=sad, 3=fear, 4=neutral, 5=surprised, 6=disgusted, 7=contemptuous)")

    args = parser.parse_args()

    # Create server
    server = GeminiDittoServer(
        cfg_pkl=args.cfg_pkl,
        data_root=args.data_root,
        source_path=args.source,
        gemini_api_key=args.api_key,
        gemini_model=args.model,
        voice_name=args.voice,
        enable_interruptions=not args.no_interruptions,
        host=args.host,
        port=args.port,
        max_size=args.max_size,
        crop_scale=args.crop_scale,
        emo=args.emo,
    )

    try:
        await server.run()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
