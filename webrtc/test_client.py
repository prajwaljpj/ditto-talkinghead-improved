#!/usr/bin/env python3
"""
WebRTC Test Client for Ditto Talking Head

Tests the WebRTC pipeline with pre-recorded audio file instead of live microphone.
Outputs the received video to a file for inspection.

Usage:
    python test_client.py --audio input.wav --image avatar.jpg --output output.mp4
"""

import asyncio
import argparse
import json
import logging
import numpy as np
import av
from fractions import Fraction
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack, AudioStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRecorder, MediaPlayer
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileAudioTrack(AudioStreamTrack):
    """
    Audio track that reads from a file and sends it as if from a microphone.
    """

    def __init__(self, audio_file_path: str, loop: bool = False):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.loop = loop
        self.container = av.open(audio_file_path, 'r')
        self.audio_stream = self.container.streams.audio[0]
        self.sample_rate = 16000  # Target sample rate for Ditto
        self.samples_per_frame = 960  # 48kHz @ 20ms (will be resampled to 16kHz)
        self._timestamp = 0
        self._resampler = None

        logger.info(f"Loaded audio file: {audio_file_path}")
        logger.info(f"  Format: {self.audio_stream.codec_context.name}")
        logger.info(f"  Sample rate: {self.audio_stream.codec_context.rate}")
        logger.info(f"  Channels: {self.audio_stream.codec_context.channels}")
        logger.info(f"  Duration: {self.container.duration / 1000000:.2f}s")

    async def recv(self):
        """Read next audio frame from file."""
        try:
            # Read next frame from file
            for packet in self.container.demux(self.audio_stream):
                for frame in packet.decode():
                    # Resample to target sample rate if needed
                    if frame.sample_rate != self.sample_rate:
                        if self._resampler is None:
                            self._resampler = av.AudioResampler(
                                format='s16',
                                layout='mono',
                                rate=self.sample_rate
                            )
                        frame = self._resampler.resample(frame)[0]

                    # Set timestamp
                    frame.pts = self._timestamp
                    frame.time_base = Fraction(1, self.sample_rate)
                    self._timestamp += frame.samples

                    return frame

            # End of file
            if self.loop:
                # Restart from beginning
                self.container.seek(0)
                return await self.recv()
            else:
                logger.info("Audio file finished")
                # Return silence
                frame = av.AudioFrame(format='s16', layout='mono', samples=self.samples_per_frame)
                frame.sample_rate = self.sample_rate
                frame.pts = self._timestamp
                frame.time_base = Fraction(1, self.sample_rate)
                for p in frame.planes:
                    p.update(bytes(p.buffer_size))
                self._timestamp += self.samples_per_frame
                return frame

        except Exception as e:
            logger.error(f"Error reading audio frame: {e}", exc_info=True)
            raise


class TestClient:
    """
    WebRTC test client that sends pre-recorded audio and receives video.
    """

    def __init__(self, server_url: str, audio_file: str, image_path: str, output_file: str, loop_audio: bool = False):
        self.server_url = server_url
        self.audio_file = audio_file
        self.image_path = image_path
        self.output_file = output_file
        self.loop_audio = loop_audio

        self.ws = None
        self.pc = None
        self.recorder = None

    async def connect(self):
        """Connect to WebRTC server."""
        logger.info(f"Connecting to {self.server_url}")

        # Create WebSocket connection
        self.ws = await websockets.connect(self.server_url)

        # Create peer connection with ICE servers
        configuration = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=['stun:stun.l.google.com:19302']),
                RTCIceServer(urls=['stun:stun1.l.google.com:19302']),
            ]
        )
        self.pc = RTCPeerConnection(configuration=configuration)

        # Set up event handlers
        @self.pc.on("icecandidate")
        async def on_icecandidate(event):
            if event.candidate:
                await self.ws.send(json.dumps({
                    'type': 'ice-candidate',
                    'candidate': {
                        'candidate': event.candidate.candidate,
                        'sdpMid': event.candidate.sdpMid,
                        'sdpMLineIndex': event.candidate.sdpMLineIndex
                    }
                }))

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received {track.kind} track")

            # Initialize recorder on first track (video or audio)
            if self.recorder is None:
                logger.info(f"Creating recorder for {self.output_file}")
                self.recorder = MediaRecorder(self.output_file)

            # Add track to recorder
            logger.info(f"Adding {track.kind} track to recorder")
            self.recorder.addTrack(track)

            # Start recorder when we have at least one track
            if not hasattr(self, '_recorder_started'):
                logger.info("Starting recorder...")
                await self.recorder.start()
                self._recorder_started = True

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state: {self.pc.iceConnectionState}")

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {self.pc.connectionState}")

            if self.pc.connectionState == "connected":
                logger.info("✓ WebRTC connected!")

            elif self.pc.connectionState in ["failed", "closed"]:
                logger.info("Connection closed, stopping...")
                await self.stop()

        # Send connect message
        await self.ws.send(json.dumps({
            'type': 'connect',
            'source': self.image_path
        }))

        # Wait for ready message
        message = json.loads(await self.ws.recv())
        logger.info(f"Received: {message}")

        if message['type'] != 'ready':
            raise Exception(f"Expected 'ready', got '{message['type']}'")

        # Add audio track from file (for sending)
        logger.info(f"Adding audio track from {self.audio_file}")
        audio_track = FileAudioTrack(self.audio_file, loop=self.loop_audio)
        self.pc.addTrack(audio_track)

        # Add transceiver for receiving video
        self.pc.addTransceiver('video', direction='recvonly')

        # Note: We're sending audio (from addTrack above) and receiving audio echo
        # The audio track we added will automatically set up a sendrecv transceiver

        # Create and send offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        await self.ws.send(json.dumps({
            'type': 'offer',
            'sdp': self.pc.localDescription.sdp
        }))

        # Wait for answer
        while True:
            message = json.loads(await self.ws.recv())

            if message['type'] == 'answer':
                answer = RTCSessionDescription(sdp=message['sdp'], type='answer')
                await self.pc.setRemoteDescription(answer)
                logger.info("✓ Received answer, connection established")
                break

            elif message['type'] == 'ice-candidate':
                candidate = RTCIceCandidate(
                    candidate=message['candidate']['candidate'],
                    sdpMid=message['candidate']['sdpMid'],
                    sdpMLineIndex=message['candidate']['sdpMLineIndex']
                )
                await self.pc.addIceCandidate(candidate)

            elif message['type'] == 'error':
                raise Exception(f"Server error: {message['message']}")

    async def run(self, duration: float = None):
        """Run the test for specified duration (or until audio file ends)."""
        logger.info(f"Running test...")

        if duration:
            logger.info(f"Will run for {duration} seconds")
            await asyncio.sleep(duration)
        else:
            # Calculate duration from audio file
            container = av.open(self.audio_file, 'r')
            duration = container.duration / 1000000  # Convert microseconds to seconds
            container.close()

            logger.info(f"Will run for {duration:.2f} seconds (audio file duration)")
            await asyncio.sleep(duration + 2)  # Add 2 seconds buffer

        logger.info("Test completed")

    async def stop(self):
        """Stop the test and close connections."""
        logger.info("Stopping...")

        if self.recorder:
            await self.recorder.stop()
            logger.info(f"✓ Video saved to {self.output_file}")

        if self.pc:
            await self.pc.close()

        if self.ws:
            await self.ws.close()


async def main():
    parser = argparse.ArgumentParser(description='WebRTC Test Client for Ditto Talking Head')
    parser.add_argument('--audio', required=True, help='Path to audio file (wav, mp3, etc.)')
    parser.add_argument('--image', required=True, help='Path to avatar image')
    parser.add_argument('--output', default='test_output.mp4', help='Output video file')
    parser.add_argument('--server', default='ws://localhost:8080', help='WebSocket server URL')
    parser.add_argument('--duration', type=float, help='Test duration in seconds (default: audio file length)')
    parser.add_argument('--loop', action='store_true', help='Loop audio file')

    args = parser.parse_args()

    logger.info("=== Ditto WebRTC Test Client ===")
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Server: {args.server}")

    client = TestClient(
        server_url=args.server,
        audio_file=args.audio,
        image_path=args.image,
        output_file=args.output,
        loop_audio=args.loop
    )

    try:
        await client.connect()
        await client.run(duration=args.duration)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await client.stop()


if __name__ == '__main__':
    asyncio.run(main())
