"""
WebSocket streaming router for real-time talking head generation.
"""

import asyncio
import json
import time
import base64
import queue
from pathlib import Path
from typing import Optional
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from api.config import get_settings, Settings
from api.models import (
    SessionStatus,
    SessionResponse,
    WSStatusMessage,
    WSFrameMessage,
    WSProgressMessage,
    WSErrorMessage,
    WSCompleteMessage,
)
from api.services import (
    get_session_manager,
    get_inference_service,
    SessionManager,
    StreamingInferenceService,
    base64_to_image,
    base64_to_audio,
    image_to_base64,
)
from core.utils.logging_config import get_logger, CorrelationIdContext

logger = get_logger(__name__)
router = APIRouter(prefix="/stream", tags=["streaming"])


async def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path, fps: int = 25) -> None:
    """
    Add audio track to video file using ffmpeg.

    Args:
        video_path: Path to video file (without audio)
        audio_path: Path to audio file
        output_path: Path for output video with audio
        fps: Frame rate for output video
    """
    import subprocess

    logger.info(f"Adding audio to video: {video_path} + {audio_path} -> {output_path} at {fps} fps")

    # Use ffmpeg to mux video and audio
    # Note: Using -c:v copy to avoid re-encoding, so frame rate comes from source video
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-fflags", "+genpts",  # Generate presentation timestamps
        "-i", str(video_path),  # Input video
        "-i", str(audio_path),  # Input audio
        "-c:v", "copy",  # Copy video codec (no re-encoding)
        "-c:a", "aac",  # Encode audio as AAC
        "-b:a", "192k",  # Audio bitrate
        "-map", "0:v:0",  # Map video from first input
        "-map", "1:a:0",  # Map audio from second input
        "-movflags", "+faststart",  # Enable fast start for web playback
        str(output_path)
    ]

    def run_ffmpeg():
        try:
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                logger.error(f"ffmpeg stdout: {result.stdout}")
                raise RuntimeError(f"Failed to add audio to video: {result.stderr}")

            logger.info(f"ffmpeg output: {result.stderr}")  # ffmpeg outputs to stderr
            logger.info(f"Successfully added audio to video: {output_path}")

            # Verify the output file
            if output_path.exists():
                file_size = output_path.stat().st_size
                logger.info(f"Output file created: {output_path}, size: {file_size} bytes")
            else:
                logger.error(f"Output file not created: {output_path}")

            return result

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timeout")
            raise RuntimeError("Audio muxing timeout")
        except Exception as e:
            logger.exception(f"Error adding audio to video")
            raise

    # Run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_ffmpeg)


async def monitor_frames(
    websocket: WebSocket,
    inf_service,
    session_id: str,
    stop_event: asyncio.Event
):
    """
    Monitor the writer queue and send frames to the WebSocket client.
    """
    import cv2
    frame_count = 0
    start_time = time.time()

    try:
        writer_queue = inf_service.get_writer_queue()
        if not writer_queue:
            logger.warning("No writer queue available for frame monitoring")
            return

        logger.info(f"Starting frame monitoring for session {session_id}")
        logger.info(f"Writer queue type: {type(writer_queue)}, maxsize: {writer_queue.maxsize if hasattr(writer_queue, 'maxsize') else 'N/A'}")

        while not stop_event.is_set():
            try:
                # Check for frames in queue (non-blocking with timeout)
                try:
                    frame = await asyncio.get_event_loop().run_in_executor(
                        None, writer_queue.get, True, 0.5
                    )
                except queue.Empty:
                    # No frame available yet, continue waiting
                    continue

                if frame is None:
                    # None signals end of stream
                    logger.info(f"End of frame stream for session {session_id}")
                    break

                frame_count += 1

                # Convert frame to JPEG (frame is in RGB, cv2 expects BGR)
                frame_bgr = frame[..., ::-1]  # RGB to BGR conversion
                _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Calculate stats
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Send frame to client
                await websocket.send_json({
                    "type": "frame",
                    "data": f"data:image/jpeg;base64,{frame_b64}",
                    "frame_id": frame_count,
                    "timestamp": time.time()
                })

                # Send progress every 10 frames
                if frame_count % 10 == 0:
                    await websocket.send_json({
                        "type": "progress",
                        "processed": frame_count,
                        "fps": fps,
                        "latency_ms": 0
                    })
                    logger.debug(f"Sent {frame_count} frames, {fps:.1f} FPS")

            except Exception as e:
                logger.exception(f"Error in frame processing loop for session {session_id}")
                break

        logger.info(f"Frame monitoring complete for session {session_id}. Sent {frame_count} frames")

    except Exception as e:
        logger.exception(f"Fatal frame monitoring error for session {session_id}")


@router.websocket("/ws")
async def websocket_streaming_endpoint(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings)
):
    """
    WebSocket endpoint for real-time streaming inference.

    Protocol:
        Client -> Server:
            {"type": "init", "source": "<base64>", "config": {...}}
            {"type": "audio_chunk", "data": "<base64>", "timestamp": 123.45}
            {"type": "stop"}

        Server -> Client:
            {"type": "status", "message": "..."}
            {"type": "frame", "data": "<base64>", "frame_id": 1, "timestamp": ...}
            {"type": "progress", "processed": 10, "fps": 25.5, "latency_ms": 45.2}
            {"type": "error", "message": "...", "code": "..."}
            {"type": "complete", "total_frames": 100, "duration_seconds": 4.5}
    """
    await websocket.accept()

    session_id: Optional[str] = None
    session_manager = get_session_manager(settings.max_concurrent_sessions)
    inference_service = get_inference_service(settings.cfg_pkl, settings.data_root)

    source_path: Optional[Path] = None
    output_path: Optional[Path] = None

    frame_count = 0
    start_time = time.time()
    last_progress_time = start_time

    try:
        logger.info("WebSocket connection established")

        # Send initial status
        await websocket.send_json({
            "type": "status",
            "message": "Connected. Send 'init' message with source image/video."
        })

        # Main message loop
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=settings.ws_heartbeat_interval
                )
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                continue

            msg_type = message.get("type")

            # ===== INIT MESSAGE =====
            if msg_type == "init":
                try:
                    # Create session
                    session = await session_manager.create_session(
                        config=message.get("config", {})
                    )
                    session_id = session.session_id

                    with CorrelationIdContext(session_id):
                        logger.info(f"Initializing session {session_id}")

                        # Save source image/video
                        source_b64 = message.get("source")
                        if not source_b64:
                            raise ValueError("Missing 'source' in init message")

                        source_image = base64_to_image(source_b64)
                        source_path = settings.temp_dir / f"{session_id}_source.png"
                        source_image.save(source_path)

                        logger.info(f"Source saved: {source_path}")

                        # Setup output path
                        output_path = settings.output_dir / f"{session_id}_output.mp4"

                        # Create inference service for this session
                        inf_service = await inference_service.create_session(session_id)

                        # Parse config
                        config = message.get("config", {})
                        logger.info(f"Received config: {config}")
                        setup_kwargs = {
                            "online_mode": True,
                            # N_d will be set via setup_Nd() after audio is loaded
                            "max_size": config.get("max_size", 1920),
                            "crop_scale": config.get("crop_scale", 2.3),
                            "emo": config.get("emo", 4),
                            "sampling_timesteps": config.get("sampling_timesteps", 50),
                            "fps": config.get("fps", 25),
                        }
                        logger.info(f"Setup kwargs FPS: {setup_kwargs['fps']}")

                        # Check if audio is provided (full audio processing)
                        audio_b64 = message.get("audio")

                        if audio_b64:
                            # Full audio mode: process immediately
                            logger.info(f"Processing with full audio for session {session_id}")

                            # Decode audio
                            import librosa
                            import io
                            import soundfile as sf
                            audio_bytes = base64.b64decode(audio_b64.split(',')[1] if ',' in audio_b64 else audio_b64)

                            # Save original audio file for later muxing
                            audio_path = settings.temp_dir / f"{session_id}_audio.wav"
                            with open(audio_path, 'wb') as f:
                                f.write(audio_bytes)
                            logger.info(f"Audio saved: {audio_path}")

                            # Load and process audio for inference
                            audio, sr = sf.read(io.BytesIO(audio_bytes))
                            if sr != 16000:
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                            if audio.ndim > 1:
                                audio = audio.mean(axis=1)  # Convert stereo to mono
                            audio = audio.astype(np.float32)

                            # Calculate N_d (frame count based on audio duration and fps)
                            import math
                            fps = config.get("fps", 25)
                            num_f = math.ceil(len(audio) / 16000 * fps)
                            logger.info(f"Audio duration: {len(audio)/16000:.2f}s, FPS: {fps}, Total frames: {num_f}")

                            # Video without audio will be written to temp file
                            video_no_audio_path = settings.temp_dir / f"{session_id}_video_no_audio.mp4"

                            # Setup session WITHOUT N_d (will be set via setup_Nd() below)
                            await inf_service.setup_session(
                                session_id,
                                str(source_path),
                                str(video_no_audio_path),
                                **setup_kwargs
                            )

                            # CRITICAL: Call setup_Nd() AFTER setup() with correct frame count
                            # This is what inference.py does at line 43
                            # N_d must ONLY be set here, not in setup_kwargs above!
                            await inf_service.setup_Nd(
                                session_id,
                                N_d=num_f,
                                fps=fps
                            )

                            await websocket.send_json({
                                "type": "status",
                                "message": f"Processing audio ({len(audio)/16000:.1f}s)...",
                                "session_id": session_id
                            })

                            # Start frame monitoring and audio processing concurrently
                            await session_manager.update_session_status(session_id, "streaming")

                            stop_event = asyncio.Event()

                            # Create concurrent tasks
                            monitor_task = asyncio.create_task(
                                monitor_frames(websocket, inf_service, session_id, stop_event)
                            )

                            # Process audio (this will feed frames to the writer queue)
                            await inf_service.process_full_audio(session_id, audio, chunksize=(3, 5, 2))

                            # Wait for all frames to be generated and written
                            logger.info(f"Waiting for all {num_f} frames to be generated...")
                            await asyncio.sleep(2)  # Give threads time to process queue

                            # Wait for writer queue to be mostly empty (frames being written)
                            max_wait = 60  # Max 60 seconds
                            wait_time = 0
                            while wait_time < max_wait:
                                queue_size = inf_service.sdk.writer_queue.qsize() if inf_service.sdk else 0
                                if queue_size == 0:
                                    logger.info("Writer queue empty, waiting for final frames...")
                                    await asyncio.sleep(1)  # Give writer time to finish
                                    break
                                logger.debug(f"Writer queue size: {queue_size}, waiting...")
                                await asyncio.sleep(0.5)
                                wait_time += 0.5

                            # Close session (this will finish writing)
                            await inf_service.close_session(session_id)

                            # Signal frame monitor to stop and wait for it
                            stop_event.set()
                            await monitor_task

                            # Add audio to video
                            await websocket.send_json({
                                "type": "status",
                                "message": "Adding audio to video...",
                                "session_id": session_id
                            })

                            # The video file has .tmp.mp4 extension
                            video_no_audio_tmp = Path(str(video_no_audio_path) + ".tmp.mp4")
                            if not video_no_audio_tmp.exists():
                                logger.warning(f"Video temp file not found at {video_no_audio_tmp}, checking without .tmp")
                                if video_no_audio_path.exists():
                                    video_no_audio_tmp = video_no_audio_path

                            logger.info(f"Video file before muxing: {video_no_audio_tmp}")
                            logger.info(f"Audio file: {audio_path}")
                            logger.info(f"FPS for muxing: {fps}")

                            # Mux audio with video
                            await add_audio_to_video(
                                video_no_audio_tmp,
                                audio_path,
                                output_path,
                                fps=fps
                            )

                            # Clean up temp files
                            try:
                                if video_no_audio_tmp.exists():
                                    video_no_audio_tmp.unlink()
                                if audio_path.exists():
                                    audio_path.unlink()
                                logger.info(f"Cleaned up temp files for session {session_id}")
                            except Exception as e:
                                logger.warning(f"Error cleaning up temp files: {e}")

                            await websocket.send_json({
                                "type": "complete",
                                "message": "Processing complete",
                                "session_id": session_id,
                                "output_path": str(output_path)
                            })

                            await session_manager.update_session_status(session_id, "completed")
                            logger.info(f"Session {session_id} completed")

                        else:
                            # Online streaming mode: wait for audio chunks
                            # Setup session
                            await inf_service.setup_session(
                                session_id,
                                str(source_path),
                                str(output_path),
                                **setup_kwargs
                            )

                            await session_manager.update_session_status(session_id, "ready")
                            session.source_path = str(source_path)
                            session.output_path = str(output_path)

                            await websocket.send_json({
                                "type": "status",
                                "message": "Session initialized. Ready for audio chunks.",
                                "session_id": session_id
                            })

                            logger.info(f"Session {session_id} ready for streaming")

                except Exception as e:
                    logger.exception("Error during initialization")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "code": "INIT_ERROR"
                    })
                    if session_id:
                        await session_manager.update_session_status(
                            session_id, "error", str(e)
                        )
                    break

            # ===== AUDIO CHUNK MESSAGE =====
            elif msg_type == "audio_chunk":
                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not initialized. Send 'init' first.",
                        "code": "NOT_INITIALIZED"
                    })
                    continue

                with CorrelationIdContext(session_id):
                    try:
                        await session_manager.update_session_status(session_id, "streaming")

                        # Decode audio
                        audio_b64 = message.get("data")
                        if not audio_b64:
                            raise ValueError("Missing 'data' in audio_chunk message")

                        audio_chunk = base64_to_audio(audio_b64)
                        timestamp = message.get("timestamp", time.time())

                        # Process audio chunk
                        inf_service_inst = await inference_service.get_session(session_id)
                        if not inf_service_inst:
                            raise RuntimeError("Session not found in inference service")

                        await inf_service_inst.process_audio_chunk(
                            session_id,
                            audio_chunk,
                            chunksize=(3, 5, 2)
                        )

                        # Check for new frames in writer queue
                        # Note: In a real implementation, we'd need a frame monitor thread
                        # For now, we just acknowledge receipt
                        frame_count += 1
                        await session_manager.increment_frames(session_id, 1)

                        # Send progress update every second
                        current_time = time.time()
                        if current_time - last_progress_time >= 1.0:
                            elapsed = current_time - start_time
                            fps = frame_count / elapsed if elapsed > 0 else 0

                            await websocket.send_json({
                                "type": "progress",
                                "processed": frame_count,
                                "fps": round(fps, 2),
                                "latency_ms": round((current_time - timestamp) * 1000, 2)
                            })

                            last_progress_time = current_time

                        logger.debug(f"Processed audio chunk {frame_count} for session {session_id}")

                    except Exception as e:
                        logger.exception(f"Error processing audio chunk")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e),
                            "code": "PROCESSING_ERROR"
                        })

            # ===== STOP MESSAGE =====
            elif msg_type == "stop":
                if session_id:
                    with CorrelationIdContext(session_id):
                        logger.info(f"Stop requested for session {session_id}")

                        try:
                            # Close inference session
                            inf_service_inst = await inference_service.get_session(session_id)
                            if inf_service_inst:
                                await inf_service_inst.close_session(session_id)

                            # Remove from inference service
                            await inference_service.remove_session(session_id)

                            # Update session status
                            await session_manager.update_session_status(session_id, "completed")

                            # Send completion message
                            total_time = time.time() - start_time
                            await websocket.send_json({
                                "type": "complete",
                                "total_frames": frame_count,
                                "duration_seconds": round(total_time, 2)
                            })

                            logger.info(f"Session {session_id} completed: {frame_count} frames in {total_time:.2f}s")

                        except Exception as e:
                            logger.exception("Error during stop")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e),
                                "code": "STOP_ERROR"
                            })

                # Close WebSocket
                await websocket.close()
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "code": "UNKNOWN_MESSAGE_TYPE"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

    except Exception as e:
        logger.exception("Unexpected error in WebSocket handler")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "code": "INTERNAL_ERROR"
            })
        except:
            pass

    finally:
        # Cleanup
        if session_id:
            logger.info(f"Cleaning up session {session_id}")
            try:
                # Remove inference session
                await inference_service.remove_session(session_id)
                # Remove from session manager
                await session_manager.remove_session(session_id)

                # Cleanup temp files
                if source_path and source_path.exists():
                    source_path.unlink()
            except Exception as e:
                logger.exception(f"Error during cleanup for session {session_id}")


@router.get("/sessions", response_model=dict)
async def list_sessions(
    session_manager: SessionManager = Depends(lambda: get_session_manager())
):
    """List all active sessions."""
    sessions = await session_manager.get_all_sessions()
    return {
        "sessions": [
            {
                "session_id": sid,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "frames_processed": s.frames_processed
            }
            for sid, s in sessions.items()
        ],
        "total": len(sessions),
        "active": session_manager.get_active_session_count()
    }


@router.get("/stats", response_model=dict)
async def get_streaming_stats(
    session_manager: SessionManager = Depends(lambda: get_session_manager())
):
    """Get streaming statistics."""
    stats = await session_manager.get_session_stats()
    return stats
