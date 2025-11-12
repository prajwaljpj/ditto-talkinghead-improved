"""
Simple Audio Passthrough Server
Just echoes audio back - no Ditto processing
"""

import asyncio
import json
import logging
from pathlib import Path
from fractions import Fraction
import numpy as np
import websockets
from websockets.legacy.server import WebSocketServerProtocol

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.mediastreams import AudioStreamTrack
import av

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PassthroughAudioTrack(AudioStreamTrack):
    """
    Simply echoes received audio back
    """

    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.sample_rate = 48000
        self._timestamp = 0
        self._frame_count = 0

    def add_audio_frame(self, frame: av.AudioFrame):
        """Add received audio frame to queue"""
        try:
            self.audio_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # Drop if full

    async def recv(self):
        """Send audio frames back"""
        try:
            # Get audio from queue (with short timeout)
            frame = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)

            # Log first frame
            if self._frame_count == 0:
                logger.info(f"First passthrough frame: {frame.format.name}, {frame.sample_rate}Hz, {frame.samples} samples")

            self._frame_count += 1
            if self._frame_count % 100 == 0:
                logger.info(f"Passthrough frames: {self._frame_count}, queue size: {self.audio_queue.qsize()}")

            return frame

        except asyncio.TimeoutError:
            # Send silence if no audio available
            frame = av.AudioFrame(format='s16', layout='stereo', samples=960)
            frame.sample_rate = self.sample_rate
            frame.pts = self._timestamp
            frame.time_base = Fraction(1, self.sample_rate)

            for p in frame.planes:
                p.update(bytes(p.buffer_size))

            self._timestamp += 960
            return frame


class PassthroughSession:
    def __init__(self, websocket: WebSocketServerProtocol):
        self.websocket = websocket
        self.pc: RTCPeerConnection = None
        self.audio_track: PassthroughAudioTrack = None

    async def handle_connect(self, message: dict):
        """Handle connect message"""
        logger.info("Client connected for passthrough test")

        # Create peer connection
        self.pc = RTCPeerConnection()

        # Create passthrough audio track
        self.audio_track = PassthroughAudioTrack()

        # Handle ICE candidates
        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await self.send_message({
                    "type": "ice-candidate",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    }
                })

        # Handle incoming audio track
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received track: {track.kind}")

            if track.kind == "audio":
                logger.info("Starting audio passthrough")

                # Echo audio frames back
                while True:
                    try:
                        frame = await track.recv()
                        self.audio_track.add_audio_frame(frame)
                    except Exception as e:
                        logger.info(f"Audio track ended: {e}")
                        break

        # Add audio track to send back
        self.pc.addTrack(self.audio_track)

        await self.send_message({"type": "ready"})

    async def handle_offer(self, message: dict):
        """Handle WebRTC offer"""
        offer = RTCSessionDescription(sdp=message["sdp"], type="offer")
        await self.pc.setRemoteDescription(offer)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        await self.send_message({
            "type": "answer",
            "sdp": self.pc.localDescription.sdp
        })

        logger.info("WebRTC handshake complete")

    async def handle_ice_candidate(self, message: dict):
        """Handle ICE candidate"""
        try:
            candidate = RTCIceCandidate(
                candidate=message["candidate"]["candidate"],
                sdpMid=message["candidate"]["sdpMid"],
                sdpMLineIndex=message["candidate"]["sdpMLineIndex"]
            )
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Error handling ICE candidate: {e}")

    async def send_message(self, message: dict):
        """Send JSON message to client"""
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def send_error(self, error_message: str):
        """Send error message"""
        await self.send_message({"type": "error", "message": error_message})

    async def cleanup(self):
        """Cleanup resources"""
        if self.pc:
            await self.pc.close()


class PassthroughServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8081):
        self.host = host
        self.port = port

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle client connection"""
        session = PassthroughSession(websocket)

        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "connect":
                    await session.handle_connect(data)
                elif msg_type == "offer":
                    await session.handle_offer(data)
                elif msg_type == "ice-candidate":
                    await session.handle_ice_candidate(data)
                else:
                    logger.warning(f"Unknown message type: {msg_type}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
            await session.send_error(str(e))
        finally:
            await session.cleanup()

    async def run(self):
        """Start the server"""
        logger.info(f"Starting Passthrough Server on {self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Server running. Connect clients to ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Audio Passthrough Test Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to bind to')
    args = parser.parse_args()

    server = PassthroughServer(host=args.host, port=args.port)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
