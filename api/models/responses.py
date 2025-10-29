"""
Pydantic models for API responses.
"""

from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class SessionStatus(str, Enum):
    """Session status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class SessionResponse(BaseModel):
    """Response for session creation/status."""
    session_id: str = Field(..., description="Unique session identifier")
    status: SessionStatus = Field(..., description="Current session status")
    created_at: datetime = Field(..., description="Session creation timestamp")
    message: Optional[str] = Field(None, description="Status message or error")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    backend: str = Field(..., description="Inference backend (pytorch/tensorrt)")
    gpu_available: bool = Field(..., description="GPU availability")
    active_sessions: int = Field(..., description="Number of active sessions")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    backend: str
    loaded: bool
    device: str


class ModelsResponse(BaseModel):
    """Response for models endpoint."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    active_backend: str = Field(..., description="Currently active backend")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class WebSocketMessage(BaseModel):
    """Base WebSocket message."""
    type: str = Field(..., description="Message type")


class WSInitMessage(WebSocketMessage):
    """Initialize WebSocket session."""
    type: Literal["init"] = "init"
    source: str = Field(..., description="Base64-encoded source image/video")
    audio: Optional[str] = Field(None, description="Base64-encoded audio file (optional, for full audio processing)")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Stream configuration")


class WSAudioChunkMessage(WebSocketMessage):
    """Audio chunk message."""
    type: Literal["audio_chunk"] = "audio_chunk"
    data: str = Field(..., description="Base64-encoded audio data (PCM, 16kHz, int16)")
    timestamp: float = Field(..., description="Timestamp in seconds")


class WSStopMessage(WebSocketMessage):
    """Stop streaming message."""
    type: Literal["stop"] = "stop"


class WSStatusMessage(WebSocketMessage):
    """Status update message."""
    type: Literal["status"] = "status"
    message: str = Field(..., description="Status message")
    session_id: Optional[str] = Field(None, description="Session ID")


class WSFrameMessage(WebSocketMessage):
    """Frame data message."""
    type: Literal["frame"] = "frame"
    data: str = Field(..., description="Base64-encoded frame (JPEG)")
    frame_id: int = Field(..., description="Frame sequence number")
    timestamp: float = Field(..., description="Frame timestamp")


class WSProgressMessage(WebSocketMessage):
    """Progress update message."""
    type: Literal["progress"] = "progress"
    processed: int = Field(..., description="Number of frames processed")
    fps: float = Field(..., description="Current processing FPS")
    latency_ms: float = Field(..., description="Average latency in milliseconds")


class WSErrorMessage(WebSocketMessage):
    """Error message."""
    type: Literal["error"] = "error"
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class WSCompleteMessage(WebSocketMessage):
    """Completion message."""
    type: Literal["complete"] = "complete"
    total_frames: int = Field(..., description="Total frames generated")
    duration_seconds: float = Field(..., description="Total processing duration")
