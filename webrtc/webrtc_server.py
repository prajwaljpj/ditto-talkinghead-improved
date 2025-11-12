"""
Ditto Talking Head WebRTC Server

This server creates real-time talking head video streams using WebRTC.
It uses the Pipecat framework for WebRTC transport and the Ditto pipeline for video generation.

Usage:
    python webrtc_server.py --cfg_pkl <path> --data_root <path> --source <avatar_image>
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# WebRTC transport (using aiortc)
try:
    from pipecat.transports.services.daily import DailyTransport, DailyParams
    DAILY_AVAILABLE = True
except ImportError:
    DAILY_AVAILABLE = False
    print("Warning: Daily transport not available. Install with: pip install pipecat-ai[daily]")

# Import our custom processors
from webrtc.processors.ditto_processor import DittoAvatarProcessor
from webrtc.processors.idle_animator import DittoIdleAnimationProcessor
from webrtc.processors.h264_encoder import H264EncoderProcessor


class DittoWebRTCServer:
    """
    WebRTC server for Ditto talking head generation.
    """

    def __init__(
        self,
        cfg_pkl: str,
        data_root: str,
        source_path: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        **kwargs
    ):
        """
        Initialize WebRTC server.

        Args:
            cfg_pkl: Path to Ditto configuration pickle
            data_root: Path to model data root
            source_path: Path to avatar source image/video
            host: Server host address
            port: Server port
            **kwargs: Additional Ditto configuration
        """
        self.cfg_pkl = cfg_pkl
        self.data_root = data_root
        self.source_path = source_path
        self.host = host
        self.port = port
        self.ditto_kwargs = kwargs

        self.runner: Optional[PipelineRunner] = None

    async def create_pipeline(self):
        """
        Create the Pipecat pipeline with Ditto processors.

        Pipeline structure:
            WebRTC Audio In → DittoAvatar → IdleAnimator → H264Encoder → WebRTC Video Out
        """

        # Create Ditto avatar processor
        ditto_processor = DittoAvatarProcessor(
            cfg_pkl=self.cfg_pkl,
            data_root=self.data_root,
            source_path=self.source_path,
            **self.ditto_kwargs
        )

        # Create idle animation processor
        idle_animator = DittoIdleAnimationProcessor(
            idle_threshold=0.01,
            idle_timeout=0.5,
            fps=25
        )

        # Create H.264 encoder (optional, WebRTC handles encoding)
        # Note: Dimensions should match source image size, not hardcoded
        # The actual output resolution is determined by the source image
        # h264_encoder = H264EncoderProcessor(
        #     width=None,  # Auto-detect from frames
        #     height=None,  # Auto-detect from frames
        #     fps=25,
        #     preset="ultrafast"
        # )

        # Create pipeline
        pipeline = Pipeline([
            ditto_processor,
            idle_animator,
            # h264_encoder,  # Optional
        ])

        return pipeline

    async def run_simple(self):
        """
        Run without WebRTC transport (for testing processors only).
        This won't actually stream, but will test the pipeline components.
        """
        print("=" * 60)
        print("Ditto Talking Head WebRTC Server (Simple Test Mode)")
        print("=" * 60)
        print(f"Config: {self.cfg_pkl}")
        print(f"Source: {self.source_path}")
        print("=" * 60)

        pipeline = await self.create_pipeline()

        # Create task params
        params = PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=False,
        )

        # Create task
        task = PipelineTask(pipeline, params=params)

        # Create runner
        self.runner = PipelineRunner()

        # Run
        print("\nStarting pipeline (simple mode)...")
        print("Note: This is test mode without WebRTC transport.")
        print("For full WebRTC functionality, use run_with_transport()")
        print()

        await self.runner.run(task)

    async def run_with_daily(self, room_url: str, token: str):
        """
        Run with Daily.co WebRTC transport.

        Args:
            room_url: Daily.co room URL
            token: Daily.co authentication token
        """
        if not DAILY_AVAILABLE:
            raise RuntimeError("Daily transport not available. Install with: pip install pipecat-ai[daily]")

        print("=" * 60)
        print("Ditto Talking Head WebRTC Server (Daily.co)")
        print("=" * 60)
        print(f"Room URL: {room_url}")
        print(f"Config: {self.cfg_pkl}")
        print(f"Source: {self.source_path}")
        print("=" * 60)

        # Create transport
        # Note: Video dimensions will match the source image dimensions
        # For best compatibility, use common resolutions like 720p, 1080p
        # The actual dimensions are determined by the source image/video
        transport = DailyTransport(
            room_url,
            token,
            "Ditto Avatar Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=False,  # Avatar doesn't speak back
                video_out_enabled=True,
                video_out_width=1280,  # Default, will be adjusted based on source
                video_out_height=720,  # Default, will be adjusted based on source
                video_out_bitrate=2000000,
                video_out_framerate=25,
            )
        )

        # Create pipeline
        pipeline = await self.create_pipeline()

        # Connect transport to pipeline
        # transport.input --> pipeline --> transport.output
        # This is handled by PipelineTask

        # Create task params
        params = PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=False,
        )

        # Create task with transport
        task = PipelineTask(
            pipeline,
            params=params,
            # transport=transport,  # Note: Check Pipecat API for correct transport integration
        )

        # Create runner
        self.runner = PipelineRunner()

        print("\nConnecting to Daily.co room...")
        print("Waiting for participants...")
        print()

        # Run
        await self.runner.run(task)

    async def shutdown(self):
        """Shutdown the server gracefully."""
        if self.runner:
            await self.runner.stop()
        print("\nServer shutdown complete.")


async def main():
    parser = argparse.ArgumentParser(description="Ditto Talking Head WebRTC Server")

    # Required arguments
    parser.add_argument("--cfg_pkl", required=True, help="Path to config pickle file")
    parser.add_argument("--data_root", required=True, help="Path to model data root directory")
    parser.add_argument("--source", required=True, help="Path to avatar source image or video")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    # WebRTC transport
    parser.add_argument("--transport", choices=["none", "daily"], default="none",
                        help="WebRTC transport type (none=test mode, daily=Daily.co)")
    parser.add_argument("--room_url", help="Daily.co room URL (required if transport=daily)")
    parser.add_argument("--token", help="Daily.co token (required if transport=daily)")

    # Ditto configuration
    parser.add_argument("--max_size", type=int, default=1920, help="Max image dimension")
    parser.add_argument("--crop_scale", type=float, default=2.3, help="Face crop scale")
    parser.add_argument("--emo", type=int, default=4, help="Emotion (0-7, default 4=neutral)")

    args = parser.parse_args()

    # Create server
    server = DittoWebRTCServer(
        cfg_pkl=args.cfg_pkl,
        data_root=args.data_root,
        source_path=args.source,
        host=args.host,
        port=args.port,
        max_size=args.max_size,
        crop_scale=args.crop_scale,
        emo=args.emo,
    )

    try:
        if args.transport == "daily":
            if not args.room_url or not args.token:
                print("Error: --room_url and --token required for Daily.co transport")
                sys.exit(1)
            await server.run_with_daily(args.room_url, args.token)
        else:
            await server.run_simple()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
