#!/usr/bin/env python3
"""
Local WebRTC Test Client (Localhost Only)

Simplified version for testing on the same machine without ICE/STUN complexity.
"""

import asyncio
import argparse
import json
import logging
import av
from fractions import Fraction
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, AudioStreamTrack
from aiortc.contrib.media import MediaRecorder
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileAudioTrack(AudioStreamTrack):
    """Audio track that reads from a file."""

    def __init__(self, audio_file_path: str):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.container = av.open(audio_file_path, 'r')
        self.audio_stream = self.container.streams.audio[0]
        self.sample_rate = 16000
        self._timestamp = 0
        self._resampler = None

        logger.info(f"Loaded audio: {audio_file_path}, duration: {self.container.duration / 1000000:.2f}s")

    async def recv(self):
        for packet in self.container.demux(self.audio_stream):
            for frame in packet.decode():
                if frame.sample_rate != self.sample_rate:
                    if self._resampler is None:
                        self._resampler = av.AudioResampler(format='s16', layout='mono', rate=self.sample_rate)
                    frame = self._resampler.resample(frame)[0]

                frame.pts = self._timestamp
                frame.time_base = Fraction(1, self.sample_rate)
                self._timestamp += frame.samples
                return frame

        # End - return silence
        frame = av.AudioFrame(format='s16', layout='mono', samples=960)
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self.sample_rate)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        self._timestamp += 960
        return frame


async def run_test(server_url, audio_file, image_path, output_file, duration):
    """Run the WebRTC test."""
    ws = None
    pc = None
    recorder = None

    try:
        # Connect to WebSocket
        logger.info(f"Connecting to {server_url}")
        ws = await websockets.connect(server_url)

        # Create peer connection (NO ICE servers for localhost)
        pc = RTCPeerConnection()

        # Track connection state
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {pc.connectionState}")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE state: {pc.iceConnectionState}")

        # Handle ICE candidates
        @pc.on("icecandidate")
        async def on_icecandidate(event):
            if event.candidate:
                await ws.send(json.dumps({
                    'type': 'ice-candidate',
                    'candidate': {
                        'candidate': event.candidate.candidate,
                        'sdpMid': event.candidate.sdpMid,
                        'sdpMLineIndex': event.candidate.sdpMLineIndex
                    }
                }))

        # Handle incoming tracks
        @pc.on("track")
        async def on_track(track):
            logger.info(f"Received {track.kind} track")
            nonlocal recorder

            if recorder is None:
                recorder = MediaRecorder(output_file)
                logger.info(f"Creating recorder: {output_file}")

            recorder.addTrack(track)

            if not hasattr(pc, '_recorder_started'):
                await recorder.start()
                pc._recorder_started = True
                logger.info("✓ Recorder started")

        # Send connect message
        await ws.send(json.dumps({'type': 'connect', 'source': image_path}))

        # Wait for ready
        message = json.loads(await ws.recv())
        if message['type'] != 'ready':
            raise Exception(f"Expected 'ready', got '{message['type']}'")
        logger.info("✓ Server ready")

        # Add audio track
        audio_track = FileAudioTrack(audio_file)
        pc.addTrack(audio_track)

        # Add video transceiver
        pc.addTransceiver('video', direction='recvonly')

        # Create and send offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await ws.send(json.dumps({'type': 'offer', 'sdp': pc.localDescription.sdp}))
        logger.info("✓ Sent offer")

        # Wait for answer and ICE candidates
        while True:
            message = json.loads(await ws.recv())

            if message['type'] == 'answer':
                answer = RTCSessionDescription(sdp=message['sdp'], type='answer')
                await pc.setRemoteDescription(answer)
                logger.info("✓ Received answer")
                break
            elif message['type'] == 'ice-candidate':
                candidate = RTCIceCandidate(
                    candidate=message['candidate']['candidate'],
                    sdpMid=message['candidate']['sdpMid'],
                    sdpMLineIndex=message['candidate']['sdpMLineIndex']
                )
                await pc.addIceCandidate(candidate)

        # Wait for connection
        for _ in range(50):  # Wait up to 5 seconds
            if pc.connectionState == "connected":
                logger.info("✓ WebRTC connected!")
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning("Connection did not establish quickly, but continuing...")

        # Run for specified duration
        logger.info(f"Running for {duration} seconds...")
        await asyncio.sleep(duration)

    finally:
        logger.info("Stopping...")
        if recorder:
            await recorder.stop()
            logger.info(f"✓ Saved to: {output_file}")
        if pc:
            await pc.close()
        if ws:
            await ws.close()


async def main():
    parser = argparse.ArgumentParser(description='Local WebRTC Test Client')
    parser.add_argument('--audio', required=True, help='Audio file path')
    parser.add_argument('--image', required=True, help='Avatar image path')
    parser.add_argument('--output', default='test_output.mp4', help='Output video file')
    parser.add_argument('--server', default='ws://localhost:8080', help='Server URL')
    parser.add_argument('--duration', type=float, default=10, help='Test duration (seconds)')

    args = parser.parse_args()

    logger.info("=== Local WebRTC Test ===")
    logger.info(f"Audio: {args.audio}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Output: {args.output}")

    try:
        await run_test(args.server, args.audio, args.image, args.output, args.duration)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())
